from recommender import Recommender
# from bias_als import BiasALS
import numpy as np
import pyspark as ps
import time
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as F
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

spark = (
    ps.sql.SparkSession.builder
    # .master("local[8]")
    .appName("eval_model")
    .getOrCreate()
)


class NDCG10Evaluator(object):
    """
    Implementation of NDCG scoring method
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    def __init__(self):
        pass

    def evaluate(self, predictions_df):
        predictions_df.registerTempTable("predictions_df")
        score_df = spark.sql(
        '''
        select 1 - avg(p.dcg / a.idcg) as ndcg
        from (
            select
                x.user,
                sum(x.rating / log(2, 1 + x.pred_row_num)) as dcg
            from (
                select
                    user,
                    rating,
                    row_number() OVER (
                        PARTITION BY user
                        ORDER BY prediction DESC
                    ) as pred_row_num
                from predictions_df
            ) x 
            where x.pred_row_num <= 10
            group by x.user
        ) p
        join (
            select
                x.user,
                sum(x.rating / log(2, 1 + x.actual_row_num)) as idcg
            from (
                select
                    user,
                    rating,
                    row_number() OVER (
                        PARTITION BY user
                        ORDER BY rating DESC
                    ) as actual_row_num
                from predictions_df
            ) x 
            where x.actual_row_num <= 10
            group by x.user
        ) a on a.user = p.user
        '''
        )
        return score_df.collect()[0][0]

    def isLargerBetter(self):
        return False


class NDCGEvaluator(object):
    """
    Implementation of NDCG scoring method
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    def __init__(self):
        pass

    def evaluate(self, predictions_df):
        predictions_df.registerTempTable("predictions_df")
        score_df = spark.sql(
        '''
        select 1 - avg(ndcg) as avg_ndcg
        from (
            select
                user,
                sum(dcg) / sum(idcg) as ndcg
            from (
                select
                    user,
                    rating / log(
                        2,
                        1 + row_number()
                        OVER (
                            PARTITION BY user
                            ORDER BY prediction DESC
                        )
                    ) as dcg,
                    rating / log(
                        2,
                        1 + row_number()
                        OVER (
                            PARTITION BY user
                            ORDER BY rating DESC
                        )
                    ) as idcg
                from predictions_df
            ) x
            group by user
        )
        '''
        )
        return score_df.collect()[0][0]

    def isLargerBetter(self):
        return False


class TopQuantileEvaluator(object):
    """
    Look at 5% of most highly predicted restaurants for each user.
    Return the average actual rating of those restaurants.
    """
    def __init__(self):
        pass

    def evaluate(self, predictions_df):
        predictions_df.registerTempTable("predictions_df")
        score_df = spark.sql(
            '''
            select
                5.0 - avg(p.rating) as score
            from predictions_df p
            join (
                select
                    user,
                    percentile_approx(prediction, 0.95) as 95_percentile
                from predictions_df
                group by user
            ) x on p.user = x.user and p.prediction >= x.95_percentile
            '''
        )
        return score_df.collect()[0][0]

    def isLargerBetter(self):
        return False


def cv_grid_search(train_df, test_df):
    estimator = Recommender(
        useALS=True,
        useBias=True,
        lambda_1=0.5,
        lambda_2=0.5,
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=10,
        regParam=0.7,
        maxIter=10,
        nonnegative=False
    )

    # estimator = ALS(
    #     userCol='user',
    #     itemCol='item',
    #     ratingCol='rating',
    #     rank=2,
    #     regParam=0.7,
    #     maxIter=5,
    #     nonnegative=False,
    #     coldStartStrategy='drop'
    # )

    paramGrid = (
        ParamGridBuilder()
        # .addGrid(estimator.lambda_1, [0, 0.5, 1])
        # .addGrid(estimator.lambda_2, [0, 0.5, 1])
        .addGrid(estimator.rank, [5, 10, 20, 40])
        # .addGrid(estimator.regParam, [0.001, 0.0025, 0.005, 0.00625, 0.0075, 0.00875])
        # .addGrid(estimator.regParam, [0.56, 0.625, 0.7])
        # .addGrid(estimator.regParam, [0.5, 0.6, 0.7, 0.8, 0.9])
        # .addGrid(estimator.maxIter, [5, 10, 15])
        # .addGrid(estimator.nonnegative, [True, False])
        .build()
    )

    # evaluator = RegressionEvaluator(
    #     metricName="rmse", labelCol="rating", predictionCol="prediction")

    # evaluator = TopQuantileEvaluator()

    # evaluator = NDCGEvaluator()

    evaluator = NDCG10Evaluator()

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5
    )

    # Run cross-validation, and choose the best set of parameters.
    step_start_time = time.monotonic()
    cvModel = cv.fit(train_df)
    print('Crossval done in {} seconds.'
        .format(time.monotonic() - step_start_time))

    paramMaps = cv.getEstimatorParamMaps()
    avgMetrics = cvModel.avgMetrics

    min_index = np.argmin(avgMetrics)

    for paramMap, avgMetric in zip(paramMaps, avgMetrics):
        print('Avg Metric: {}'.format(avgMetric))
        for key, value in paramMap.items():
            print('  {}: {}'.format(key, value))

    print(avgMetrics)

    print('Best Params:')
    for key, value in paramMaps[min_index].items():
        print('{} : {}'.format(key, value))
    print('Best Metric: {}'.format(avgMetrics[min_index]))

    test_metric = evaluator.evaluate(cvModel.bestModel.transform(test_df))
    print("Test Metric: {}".format(test_metric))


def get_baseline_scores(train_df, val_df, evaluator, eval_name, lambda_1, lambda_2):
    avg_rating_df = (
        train_df
        .agg(
            F.avg('rating').alias('avg_rating')
        )
    )

    # Naive model: random normal rating centered on average rating.
    train_predict_df = (
        train_df
        .crossJoin(avg_rating_df)
        .withColumn(
            'prediction',
            F.col('avg_rating') + F.randn()
        )
        .select(
            'user',
            'item',
            'rating',
            'prediction'
        )
    )

    val_predict_df = (
        val_df
        .crossJoin(avg_rating_df)
        .withColumn(
            'prediction',
            F.col('avg_rating') + F.randn()
        )
        .select(
            'user',
            'item',
            'rating',
            'prediction'
        )
    )

    naive_score_train = evaluator.evaluate(train_predict_df)
    naive_score_val = evaluator.evaluate(val_predict_df)

    print('Train Naive {} score: {}'.format(eval_name, naive_score_train))
    print('Validation Naive {} score: {}'.format(eval_name, naive_score_val))

    estimator = Recommender(
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        useALS=False,
        userCol='user',
        itemCol='item',
        ratingCol='rating'
    )

    model = estimator.fit(train_df)
    baseline_score_train = evaluator.evaluate(model.transform(train_df))
    baseline_score_val = evaluator.evaluate(model.transform(val_df))

    print('Train Baseline {} score: {}'.format(eval_name, baseline_score_train))
    print('Validation Baseline {} score: {}'.format(eval_name, baseline_score_val))

    return (
        naive_score_train, naive_score_val,
        baseline_score_train, baseline_score_val
    )


def plot_scores(train_df):

    best_rank_so_far = 76
    best_regParam_so_far = 0.7
    lambda_1 = 7
    lambda_2 = 12
    nonnegative=True
    maxIter = 15
    useBias = True
    implicitPrefs = False

    eval_name = 'NDCG10'
    evaluator = NDCG10Evaluator()

    # eval_name = 'NDCG'
    # evaluator = NDCGEvaluator()

    # eval_name = 'TopQuantileEvaluator'
    # evaluator = TopQuantileEvaluator()

    # eval_name = 'RMSE'
    # evaluator = RegressionEvaluator(
    #     metricName="rmse", labelCol="rating", predictionCol="prediction")    

    print()
    print('best_rank_so_far: {}'.format(best_rank_so_far))
    print('best_regParam_so_far: {}'.format(best_regParam_so_far))
    print('lambda_1: {}'.format(lambda_1))
    print('lambda_2: {}'.format(lambda_2))
    print('nonnegative: {}'.format(nonnegative))
    print('maxIter: {}'.format(maxIter))
    print('useBias: {}'.format(useBias))
    print('implicitPrefs: {}'.format(implicitPrefs))
    print('eval_name: {}'.format(eval_name))
    print()

    train_df, val_df = train_df.randomSplit(weights=[0.75, 0.25])

    print_counts(train_df, 'plot_scores Train')
    print_counts(val_df, 'plot_scores Validation')


    # First get baseline scores with ALS turned off
    (   naive_score_train, naive_score_val,
        baseline_score_train, baseline_score_val
    ) = (
        get_baseline_scores(
            train_df, val_df, evaluator, eval_name, lambda_1, lambda_2)
    )
    
    ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    rank_scores_train = []
    rank_scores_val =[]

    start_time = time.monotonic()

    for rank in ranks:
        step_start_time = time.monotonic()

        estimator = Recommender(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            useALS=True,
            useBias=useBias,
            userCol='user',
            itemCol='item',
            ratingCol='rating',
            rank=rank,
            regParam=best_regParam_so_far,
            maxIter=maxIter,
            nonnegative=nonnegative,
            implicitPrefs=implicitPrefs
        )

        model = estimator.fit(train_df)

        rank_scores_train.append(evaluator.evaluate(model.transform(train_df)))
        rank_scores_val.append(evaluator.evaluate(model.transform(val_df)))

        print('rank: {} train score: {} val score: {}'
            .format(
                rank,
                rank_scores_train[-1],
                rank_scores_val[-1]
            )
        )

        print('rank: {} train diff: {} val diff: {} ({} seconds)'
            .format(
                rank,
                rank_scores_train[-1] - baseline_score_train,
                rank_scores_val[-1] - baseline_score_val,
                time.monotonic() - step_start_time
            )
        )

    print('Done in {} seconds'
        .format(time.monotonic() - start_time))

    rank_scores_train = np.array(rank_scores_train)
    rank_scores_val = np.array(rank_scores_val)
    best_rank_index = np.argmin(rank_scores_val)

    print('Ranks:')
    print(ranks)
    print('Train score:')
    print(rank_scores_train)
    print('Validation score:')
    print(rank_scores_val)
    print('Train score - Baseline:')
    print(rank_scores_train - baseline_score_train)
    print('Validation score - baseline:')
    print(rank_scores_val - baseline_score_val)
    print('Best Rank: {}'.format(ranks[best_rank_index]))


    regParams = [0.1, 0.5, 1.0, 1.5, 2.0]
    regParam_scores_train = []
    regParam_scores_val =[]

    start_time = time.monotonic()

    for regParam in regParams:
        step_start_time = time.monotonic()

        estimator = Recommender(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            useALS=True,
            useBias=useBias,
            userCol='user',
            itemCol='item',
            ratingCol='rating',
            rank=best_rank_so_far,
            regParam=regParam,
            maxIter=maxIter,
            nonnegative=nonnegative,
            implicitPrefs=implicitPrefs
        )

        model = estimator.fit(train_df)

        regParam_scores_train.append(
            evaluator.evaluate(model.transform(train_df)))
        regParam_scores_val.append(
            evaluator.evaluate(model.transform(val_df)))

        print('regParam: {} train score: {} val score: {}'
            .format(
                regParam,
                regParam_scores_train[-1],
                regParam_scores_val[-1]
            )
        )

        print('regParam: {} train diff: {} val diff: {} ({} seconds)'
            .format(
                regParam,
                regParam_scores_train[-1] - baseline_score_train,
                regParam_scores_val[-1] - baseline_score_val,
                time.monotonic() - step_start_time
            )
        )

    print('Done in {} seconds'
        .format(time.monotonic() - start_time))


    regParam_scores_train = np.array(regParam_scores_train)
    regParam_scores_val = np.array(regParam_scores_val)
    best_regParam_index = np.argmin(regParam_scores_val)

    print('RegParams:')
    print(regParams)
    print('Train score:')
    print(regParam_scores_train)
    print('Validation score:')
    print(regParam_scores_val)
    print('Train score - Baseline:')
    print(regParam_scores_train - baseline_score_train)
    print('Validation score - Baseline:')
    print(regParam_scores_val - baseline_score_val)
    print('Best RegParam: {}'.format(regParams[best_regParam_index]))



    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    flat_axes = axes.flatten()

    flat_axes[0].axhline(y=naive_score_train, label='Train Naive', 
        color='green', alpha=0.5)
    flat_axes[0].axhline(y=baseline_score_train, label='Train Baseline', 
        color='purple', alpha=0.5)
    flat_axes[0].plot(ranks, rank_scores_train, label='Train Model', alpha=0.5)
    flat_axes[0].axhline(y=naive_score_val, label='Validation Naive', 
        color='black', alpha=0.5)
    flat_axes[0].axhline(y=baseline_score_val, label='Validation Baseline', 
        color='orange', alpha=0.5)
    flat_axes[0].plot(ranks, rank_scores_val, label='Validation Model', alpha=0.5)


    flat_axes[0].set_title('{} vs. Rank (regParam={})'
        .format(eval_name, best_regParam_so_far))
    flat_axes[0].set_xlabel('Rank')
    flat_axes[0].set_ylabel(eval_name)
    flat_axes[0].legend()

    flat_axes[1].axhline(y=naive_score_train, label='Train Naive', 
        color='green', alpha=0.5)
    flat_axes[1].axhline(y=baseline_score_train, label='Train Baseline', 
        color='purple', alpha=0.5)
    flat_axes[1].plot(regParams, regParam_scores_train, label='Train Model',
        alpha=0.5)
    flat_axes[1].axhline(y=naive_score_val, label='Validation Naive', 
        color='black', alpha=0.5)
    flat_axes[1].axhline(y=baseline_score_val, label='Validation Baseline', 
        color='orange', alpha=0.5)
    flat_axes[1].plot(regParams, regParam_scores_val, label='Validation Model',
        alpha=0.5)

    flat_axes[1].set_title('{} vs. regParam (Rank={})'
        .format(eval_name, best_rank_so_far))
    flat_axes[1].set_xlabel('regParam')
    flat_axes[1].set_ylabel(eval_name)
    flat_axes[1].legend()

    flat_axes[2].plot(ranks, rank_scores_train - baseline_score_train,
        label='Train Diff', alpha=0.5)
    flat_axes[2].plot(ranks, rank_scores_val - baseline_score_val,
        label='Validation Diff', alpha=0.5)
    flat_axes[2].set_title('{} - Baseline vs. Rank (regParam={})'
        .format(eval_name, best_regParam_so_far))
    flat_axes[2].set_xlabel('Rank')
    flat_axes[2].set_ylabel(eval_name)
    flat_axes[2].legend()

    flat_axes[3].plot(regParams, regParam_scores_train - baseline_score_train,
        label='Train Diff', alpha=0.5)
    flat_axes[3].plot(regParams, regParam_scores_val - baseline_score_val,
        label='Validation Diff', alpha=0.5)
    flat_axes[3].set_title('{} - Baseline vs. regParam (Rank={})'
        .format(eval_name, best_rank_so_far))
    flat_axes[3].set_xlabel('regParam')
    flat_axes[3].set_ylabel(eval_name)
    flat_axes[3].legend()

    plt.tight_layout()
    plt.show()


def eval_model(train_df, test_df):
    # estimator = Recommender(
    #     useALS=True,
    #     lambda_1=5,
    #     lambda_2=8,
    #     userCol='user',
    #     itemCol='item',
    #     ratingCol='rating',
    #     rank=2,
    #     regParam=0.7,
    #     maxIter=5,
    #     nonnegative=True
    # )

    estimator = ALS(
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=2,
        regParam=0.7,
        maxIter=5,
        nonnegative=True,
        coldStartStrategy='drop'
    )

    evaluator_rmse = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")

    evaluator_ndcg10 = NDCG10Evaluator()

    start_time = time.monotonic()
    step_start_time = time.monotonic()

    model = estimator.fit(train_df)

    print('Fit done in {} seconds.'.format(time.monotonic() - step_start_time))

    train_predictions_df = model.transform(train_df)
    
    step_start_time = time.monotonic()
    test_predictions_df = model.transform(test_df)
    # print(predictions_df.printSchema())

    for row in test_predictions_df.head(30):
        print(row)

    print('Predictions done in {} seconds.'
        .format(time.monotonic() - step_start_time))
    print('All done in {} seconds.'.format(time.monotonic() - start_time))

    print_avg_predictions(train_predictions_df, 'Train')
    print_avg_predictions(test_predictions_df, 'Test')

    train_rmse = evaluator_rmse.evaluate(train_predictions_df)
    test_rmse = evaluator_rmse.evaluate(test_predictions_df)
    print("Train RMSE: {}".format(train_rmse))
    print("Test RMSE: {}".format(test_rmse))

    train_ndcg10 = evaluator_ndcg10.evaluate(train_predictions_df)
    test_ndcg10 = evaluator_ndcg10.evaluate(test_predictions_df)
    print("Train NDCG10: {}".format(train_ndcg10))
    print("Test NDCG10: {}".format(test_ndcg10))


def print_counts(ratings_df, label):
    print('[{}] Num total ratings: {}'
        .format(label, ratings_df.count()))
    print('[{}] Num users: {}'
        .format(label, ratings_df.groupBy('user').count().count()))
    print('[{}] Num restaurants: {}'
        .format(label, ratings_df.groupBy('item').count().count()))
    print('[{}] Avg num ratings per user: {}'
        .format(label, ratings_df.groupBy('user').count().agg(F.avg('count')).collect()[0][0]))
    print('[{}] Avg num ratings per restaurant: {}'
        .format(label, ratings_df.groupBy('item').count().agg(F.avg('count')).collect()[0][0]))


def print_avg_predictions(predictions_df, label):
    result_row = (
        predictions_df
        .agg(
            F.avg('rating').alias('avg_rating'),
            F.stddev('rating').alias('stddev_rating'),
            F.avg('prediction').alias('avg_prediction'),
            F.stddev('prediction').alias('stddev_prediction')
        ).collect()[0]
    )
    print('[{} Prediction] Rating Avg: {} Stddev: {}'
        .format(label, result_row[0], result_row[1]))
    print('[{} Prediction] Prediction Avg: {} Stddev: {}'
        .format(label, result_row[2], result_row[3]))


def main():

    # Load restaurant reviews
    ratings_df = spark.read.parquet('../data/ratings')

    # Randomly split data into train and test datasets
    train_df, test_df = ratings_df.randomSplit(weights=[0.5, 0.5])

    # print(ratings_df.printSchema())
    print_counts(ratings_df, 'Total')
    print_counts(train_df, 'Train')
    print_counts(test_df, 'Test')

    cv_grid_search(train_df, test_df)
    
    # plot_scores(train_df)
    
    # eval_model(train_df, test_df)




if __name__ == '__main__':
    main()

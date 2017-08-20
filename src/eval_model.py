from recommender import Recommender
from model_evaluators import NDCG10Evaluator, NDCGEvaluator, TopQuantileEvaluator
from ratings_helper_functions import print_ratings_counts, print_avg_predictions
# from bias_als import BiasALS
import numpy as np
import pyspark as ps
import time
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')

spark = (
    ps.sql.SparkSession.builder
    .config('spark.driver.memory', '4g')
    .config('spark.executor.memory', '8g')
    # .master("local[8]")
    .appName("eval_model")
    .getOrCreate()
)

def cv_grid_search(ratings_df):

    ratings_df1, ratings_df2 = ratings_df.randomSplit(weights=[0.5, 0.5])

    # Randomly split data into train and test datasets
    train_df, test_df = ratings_df1.randomSplit(weights=[0.833, 0.167])

    print_ratings_counts(ratings_df1, 'Split 1')
    print_ratings_counts(ratings_df2, 'Split 2')
    print_ratings_counts(train_df, 'Train')
    print_ratings_counts(test_df, 'Test')

    estimator = Recommender(
        useALS=True,
        useBias=True,
        lambda_1=0.5,
        lambda_2=0.5,
        lambda_3=0,
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=128,
        regParam=0.16,
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
        .addGrid(estimator.lambda_1, [0, 0.25, 0.5, 0.75, 1])
        .addGrid(estimator.lambda_2, [0, 0.25, 0.5, 0.75, 1])
        # .addGrid(estimator.lambda_3, [0, 0.25, 0.5, 0.75, 1])
        # .addGrid(estimator.rank, [2, 4, 8, 16, 32])
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


def get_baseline_scores(train_df, val_df, evaluator, eval_name):
    stats_rating_df = (
        train_df
        .agg(
            F.avg('rating').alias('avg_rating'),
            F.stddev_samp('rating').alias('stddev_rating')
        )
    )

    stats_row = stats_rating_df.head()

    print('[plot_scores Train] Avg: {}'.format(stats_row[0]))
    print('[plot_scores Train] Std Dev: {}'.format(stats_row[1]))

    # Naive model: random normal rating centered on average rating and scaled
    # with standard deviation of training data.
    train_predict_df = (
        train_df
        .crossJoin(stats_rating_df)
        .withColumn(
            'prediction',
            F.col('avg_rating') + F.randn() * F.col('stddev_rating')
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
        .crossJoin(stats_rating_df)
        .withColumn(
            'prediction',
            F.col('avg_rating') + F.randn() * F.col('stddev_rating')
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
        lambda_1=0.0,
        lambda_2=0.0,
        lambda_3=0.0,
        useALS=False,
        useBias=True,
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

    best_rank_so_far = 250
    best_regParam_so_far = 0.001
    lambda_1 = 2
    lambda_2 = 2
    lambda_3 = 0.0
    nonnegative = False
    maxIter = 10
    useALS = True
    useBias = True
    implicitPrefs = False

    # eval_name = 'NDCG10'
    # evaluator = NDCG10Evaluator()

    # eval_name = 'NDCG'
    # evaluator = NDCGEvaluator()

    # eval_name = 'TopQuantileEvaluator'
    # evaluator = TopQuantileEvaluator()

    eval_name = 'RMSE'
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")    

    print()
    print('best_rank_so_far: {}'.format(best_rank_so_far))
    print('best_regParam_so_far: {}'.format(best_regParam_so_far))
    print('lambda_1: {}'.format(lambda_1))
    print('lambda_2: {}'.format(lambda_2))
    print('lambda_3: {}'.format(lambda_3))
    print('nonnegative: {}'.format(nonnegative))
    print('maxIter: {}'.format(maxIter))
    print('useALS: {}'.format(useALS))
    print('useBias: {}'.format(useBias))
    print('implicitPrefs: {}'.format(implicitPrefs))
    print('eval_name: {}'.format(eval_name))
    print()

    train_df, val_df = train_df.randomSplit(weights=[0.5, 0.5])

    print_ratings_counts(train_df, 'plot_scores Train')
    print_ratings_counts(val_df, 'plot_scores Validation')


    # First get baseline scores with ALS turned off
    (   naive_score_train, naive_score_val,
        baseline_score_train, baseline_score_val
    ) = (
        get_baseline_scores(
            train_df, val_df, evaluator, eval_name)
    )
    
    ranks = [1, 2, 5, 10, 25, 50, 100, 250]
    rank_scores_train = []
    rank_scores_val =[]

    start_time = time.monotonic()

    for rank in ranks:
        step_start_time = time.monotonic()

        estimator = Recommender(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            useALS=useALS,
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


        train_predictions_df = model.transform(train_df)
        val_predictions_df = model.transform(val_df)

        # print('train_predictions_df')
        # train_predictions_df.show()

        # print('val_predictions_df')
        # val_predictions_df.show()

        # exit()

        rank_scores_train.append(evaluator.evaluate(train_predictions_df))
        rank_scores_val.append(evaluator.evaluate(val_predictions_df))

        print('rank: {} train score: {} val score: {}'
            .format(
                rank,
                rank_scores_train[-1],
                rank_scores_val[-1]
            )
        )

        print('rank: {} train diff: {} val diff: {} ({} seconds)\n'
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


    regParams = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.50, 1]
    regParam_scores_train = []
    regParam_scores_val =[]

    start_time = time.monotonic()

    for regParam in regParams:
        step_start_time = time.monotonic()

        estimator = Recommender(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            useALS=useALS,
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

        train_predictions_df = model.transform(train_df)
        val_predictions_df = model.transform(val_df)

        regParam_scores_train.append(evaluator.evaluate(train_predictions_df))
        regParam_scores_val.append(evaluator.evaluate(val_predictions_df))

        print('regParam: {} train score: {} val score: {}'
            .format(
                regParam,
                regParam_scores_train[-1],
                regParam_scores_val[-1]
            )
        )

        print('regParam: {} train diff: {} val diff: {} ({} seconds)\n'
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


def eval_model(ratings_df):

    # Randomly split data into train and test datasets
    train_df, test_df = ratings_df.randomSplit(weights=[0.5, 0.5])

    print_ratings_counts(train_df, 'Train')
    print_ratings_counts(test_df, 'Test')

    estimator = Recommender(
        useALS=True,
        useBias=True,
        lambda_1=0.5,
        lambda_2=0.5,
        lambda_3=0,
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=128,
        regParam=0.16,
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
    #     nonnegative=True,
    #     coldStartStrategy='drop'
    # )

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

    print('Predictions done in {} seconds.'
        .format(time.monotonic() - step_start_time))
    print('All done in {} seconds.'.format(time.monotonic() - start_time))

    # print(predictions_df.printSchema())

    for row in test_predictions_df.head(30):
        print(row)

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


def main():

    # Load restaurant reviews
    # ratings_df = spark.read.parquet('../data/ratings')
    # ratings_df = spark.read.parquet('../data/ratings_ugt1_igt1')
    # ratings_df = spark.read.parquet('../data/ratings_ugt5_igt5')
    ratings_df = spark.read.parquet('../data/ratings_ugt10_igt10')

    # print(ratings_df.printSchema())
    print_ratings_counts(ratings_df, 'Total')

    # cv_grid_search(ratings_df)
    
    plot_scores(ratings_df)
    
    # eval_model(ratings_df)




if __name__ == '__main__':
    main()

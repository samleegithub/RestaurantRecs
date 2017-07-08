from recommender import Recommender
import numpy as np
import pyspark as ps
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# TODO: Implement a NDCG scoring method
# https://en.wikipedia.org/wiki/Discounted_cumulative_gain


def compute_score(predictions_df):
    """Look at 5% of most highly predicted restaurants for each user.
    Return the average actual rating of those restaurants.
    """
    # for each user
    g = predictions_df.groupBy('user_id')

    # detect the top_5 restaurants as predicted by your algorithm
    top_5 = g['prediction'].transform(
        lambda x: x >= x.quantile(0.95)
    )

    # return the mean of the actual score on those
    return predictions_df['stars'][top_5].mean()


def cv_grid_search(train_df, test_df):
    estimator = Recommender(
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=100,
        regParam=1,
        maxIter=10,
        nonnegative=False
    )

    paramGrid = (
        ParamGridBuilder()
        # .addGrid(estimator.rank, [20, 50, 100])
        .addGrid(estimator.regParam, [.1, .25, .5, .75, 1])
        # .addGrid(estimator.maxIter, [10])
        # .addGrid(estimator.nonnegative, [True, False])
        .build()
    )

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3
    )

    # Run cross-validation, and choose the best set of parameters.
    step_start_time = time.monotonic()
    cvModel = cv.fit(train_df)
    print('Crossval done in {} seconds.'
        .format(time.monotonic() - step_start_time))

    print(cv.getEstimatorParamMaps())
    print(cvModel.avgMetrics)

    rmse = evaluator.evaluate(cvModel.transform(test_df))
    print("Test RMSE: {}".format(rmse))


def plot_scores(train_df):
    best_rank_so_far = 6
    best_regParam_so_far = 0.5

    train_df, val_df = train_df.randomSplit(weights=[0.75, 0.25])
    print('[plot_scores] Num train ratings: {}'.format(train_df.count()))
    print('[plot_scores] Num validation ratings: {}'.format(val_df.count()))

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")

    # First get baseline scores with ALS turned off
    estimator = Recommender(
        useALS=False,
        userCol='user',
        itemCol='item',
        ratingCol='rating'
    )

    model = estimator.fit(train_df)
    baseline_score_train = evaluator.evaluate(model.transform(train_df))
    baseline_score_val = evaluator.evaluate(model.transform(val_df))

    print('Train Baseline RSME score: {}'.format(baseline_score_train))
    print('Validation Baseline RSME score: {}'.format(baseline_score_val))
    
    ranks = [1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 35, 50, 100]
    rank_scores_train = []
    rank_scores_val =[]

    start_time = time.monotonic()

    for rank in ranks:
        step_start_time = time.monotonic()

        estimator = Recommender(
            useALS=True,
            userCol='user',
            itemCol='item',
            ratingCol='rating',
            rank=rank,
            regParam=best_regParam_so_far,
            maxIter=10,
            nonnegative=False
        )

        model = estimator.fit(train_df)

        rank_scores_train.append(evaluator.evaluate(model.transform(train_df)))
        rank_scores_val.append(evaluator.evaluate(model.transform(val_df)))

        print('rank: {} train RMSE: {} val RMSE: {}'
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


    print('Ranks:')
    print(ranks)
    print('Train RMSE:')
    print(rank_scores_train)
    print('Validation RMSE:')
    print(rank_scores_val)
    print('Train RMSE - Baseline:')
    print(rank_scores_train - baseline_score_train)
    print('Validation RMSE - baseline:')
    print(rank_scores_val - baseline_score_val)


    regParams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    regParam_scores_train = []
    regParam_scores_val =[]

    start_time = time.monotonic()

    for regParam in regParams:
        step_start_time = time.monotonic()

        estimator = Recommender(
            userCol='user',
            itemCol='item',
            ratingCol='rating',
            rank=best_rank_so_far,
            regParam=regParam,
            maxIter=10,
            nonnegative=False
        )

        model = estimator.fit(train_df)

        regParam_scores_train.append(
            evaluator.evaluate(model.transform(train_df)))
        regParam_scores_val.append(
            evaluator.evaluate(model.transform(val_df)))

        print('regParam: {} train RMSE: {} val RMSE: {}'
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

    print('RegParams:')
    print(regParams)
    print('Train RMSE:')
    print(regParam_scores_train)
    print('Validation RMSE:')
    print(regParam_scores_val)
    print('Train RMSE - Baseline:')
    print(regParam_scores_train - baseline_score_train)
    print('Validation RMSE - Baseline:')
    print(regParam_scores_val - baseline_score_val)


    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    flat_axes = axes.flatten()

    # flat_axes[0].plot(ranks, rank_scores_train, label='Train', alpha=0.5)
    flat_axes[0].plot(ranks, rank_scores_val, label='Validation', alpha=0.5)
    # flat_axes[0].axhline(y=baseline_score_train, label='Train Baseline', 
    #     color='purple', alpha=0.5)
    flat_axes[0].axhline(y=baseline_score_val, label='Validation Baseline', 
        color='orange', alpha=0.5)
    flat_axes[0].set_title('RMSE vs. Rank (regParam={})'
        .format(best_regParam_so_far))
    flat_axes[0].set_xlabel('Rank')
    flat_axes[0].set_ylabel('RMSE')
    flat_axes[0].legend()

    flat_axes[1].plot(regParams, regParam_scores_train, label='Train',
        alpha=0.5)
    flat_axes[1].plot(regParams, regParam_scores_val, label='Validation',
        alpha=0.5)
    flat_axes[1].axhline(y=baseline_score_train, label='Train Baseline', 
        color='purple', alpha=0.5)
    flat_axes[1].axhline(y=baseline_score_val, label='Validation Baseline', 
        color='orange', alpha=0.5)
    flat_axes[1].set_title('RMSE vs. regParam (Rank={})'
        .format(best_rank_so_far))
    flat_axes[1].set_xlabel('regParam')
    flat_axes[1].set_ylabel('RMSE')
    flat_axes[1].legend()

    # flat_axes[2].plot(ranks, rank_scores_train - baseline_score_train,
    #     label='Train Diff')
    flat_axes[2].plot(ranks, rank_scores_val - baseline_score_val,
        label='Validation Diff')
    flat_axes[2].set_title('RMSE - Baseline vs. Rank (regParam={})'
        .format(best_regParam_so_far))
    flat_axes[2].set_xlabel('Rank')
    flat_axes[2].set_ylabel('RMSE')
    flat_axes[2].legend()

    flat_axes[3].plot(regParams, regParam_scores_train - baseline_score_train,
        label='Train Diff')
    flat_axes[3].plot(regParams, regParam_scores_val - baseline_score_val,
        label='Validation Diff')
    flat_axes[3].set_title('RMSE - Baseline vs. regParam (Rank={})'
        .format(best_regParam_so_far))
    flat_axes[3].set_xlabel('regParam')
    flat_axes[3].set_ylabel('RMSE')
    flat_axes[3].legend()

    plt.tight_layout()
    plt.show()


def eval_model(train_df, test_df):
    estimator = Recommender(
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=10,
        regParam=0.5,
        maxIter=10,
        nonnegative=False
    )

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")

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

    train_rmse = evaluator.evaluate(train_predictions_df)
    test_rmse = evaluator.evaluate(test_predictions_df)
    print("Train RMSE: {}".format(train_rmse))
    print("Test RMSE: {}".format(test_rmse))


def main():
    spark = (
        ps.sql.SparkSession.builder
        .master("local[8]")
        .appName("eval_model")
        .getOrCreate()
    )

    # Load restaurant reviews
    ratings_df = spark.read.parquet('../data/ratings_ugt10_igt10')

    # Randomly split data into train and test datasets
    train_df, test_df = ratings_df.randomSplit(weights=[0.75, 0.25])

    print(ratings_df.printSchema())
    print('Num total ratings: {}'.format(ratings_df.count()))
    print('Num train ratings: {}'.format(train_df.count()))
    print('Num test ratings: {}'.format(test_df.count()))

    # cv_grid_search(train_df, test_df)
    
    plot_scores(train_df)
    
    # eval_model(train_df, test_df)




if __name__ == '__main__':
    main()

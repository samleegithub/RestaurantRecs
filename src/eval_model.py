from recommender import Recommender
import pyspark as ps
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS


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
    print('Crossval done in {} seconds.'.format(time.monotonic() - step_start_time))

    print(cv.getEstimatorParamMaps())
    print(cvModel.avgMetrics)

    rmse = evaluator.evaluate(cvModel.transform(test_df))
    print("Test RMSE: {}".format(rmse))


def eval_model(train_df, test_df):
    estimator = Recommender(
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=100,
        regParam=1,
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

    for row in predictions_df.head(10):
        print(row)

    print('Predictions done in {} seconds.'.format(time.monotonic() - step_start_time))
    print('All done in {} seconds.'.format(time.monotonic() - start_time))

    train_rmse = evaluator.evaluate(train_predictions_df)
    test_rmse = evaluator.evaluate(test_predictions_df)
    print("Train RMSE: {}".format(train_rmse))
    print("Test RMSE: {}".format(test_rmse))


def main():
    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("eval_model")
        .getOrCreate()
    )

    # Load restaurant reviews
    reviews_df = spark.read.parquet('s3://sam.lee/restaurantrecs/data/ratings')

    # Randomly split data into train and test datasets
    train_df, test_df = reviews_df.randomSplit(weights=[0.75, 0.25])

    print(reviews_df.printSchema())
    print('Num total ratings: {}'.format(reviews_df.count()))
    print('Num train ratings: {}'.format(train_df.count()))
    print('Num test ratings: {}'.format(test_df.count()))

    cv_grid_search(train_df, test_df)

    # eval_model(train_df, test_df)


if __name__ == '__main__':
    main()

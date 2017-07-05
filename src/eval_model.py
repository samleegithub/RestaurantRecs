from resto_reco import RestaurantRecommender
import pyspark as ps
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS

spark = (
    ps.sql.SparkSession.builder
    .master("local[4]")
    .appName("eval_model")
    .getOrCreate()
)

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


def main():
    # Load restaurant reviews
    reviews_df = spark.read.parquet('../data/ratings')

    # Randomly split data into train and test datasets
    train_df, test_df = reviews_df.randomSplit(weights=[0.75, 0.25])

    print(train_df.printSchema())
    print('Num total ratings: {}'.format(reviews_df.count()))
    print('Num train ratings: {}'.format(train_df.count()))
    print('Num test ratings: {}'.format(test_df.count()))


    model = RestaurantRecommender()
    model_test = ALS(
        userCol='user_id',
        itemCol='product_id',
        ratingCol='rating',
        nonnegative=True,
        regParam=0.1
    )

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")

    start_time = time.monotonic()
    step_start_time = time.monotonic()

    model.fit(train_df)

    print('Fit done in {} seconds.'.format(time.monotonic() - step_start_time))
    
    step_start_time = time.monotonic()
    predictions_df = model.transform(test_df)
    # print(predictions_df.printSchema())

    for row in predictions_df.head(10):
        print(row)
        
    print('Predictions done in {} seconds.'.format(time.monotonic() - step_start_time))
    print('All done in {} seconds.'.format(time.monotonic() - start_time))

    rmse = evaluator.evaluate(predictions_df)
    print("Root-mean-square error = {}".format(rmse))

    exit()

    paramGrid = (
        ParamGridBuilder()
        .addGrid(model_test.rank, [10, 15])
        .build()
    )

    crossval = CrossValidator(
        estimator=model_test,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2
    )

    print(crossval.getEstimatorParamMaps())

    # Run cross-validation, and choose the best set of parameters.
    print('{}: Starting crossval...'.format(datetime.datetime.now()))
    cvModel = crossval.fit(train_df)
    print('{}: Crossval done...'.format(datetime.datetime.now()))

    print(cvModel.avgMetrics)

    bestModel = cvModel.bestModel
    print(bestModel.explainParams())


if __name__ == '__main__':
    main()

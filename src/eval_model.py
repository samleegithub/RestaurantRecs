from resto_reco import RestaurantRecommender
import pyspark as ps
import datetime

spark = (
    ps.sql.SparkSession.builder
    .master("local[8]")
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


    print('{}: Starting fit...'.format(datetime.datetime.now()))
    model = RestaurantRecommender()
    model.fit(train_df)
    print('{}: Fit done.'.format(datetime.datetime.now()))

    predictions_df = model.transform(test_df)

    print(predictions_df.printSchema())

    for row in predictions_df.head(10):
        print(row)


if __name__ == '__main__':
    main()

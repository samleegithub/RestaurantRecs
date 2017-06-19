from RestaurantRecommender import RestaurantRecommender
import pyspark as ps
from pyspark.ml.feature import StringIndexer

def main():
    spark = (
        ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("resto-reco")
        .getOrCreate()
    )

    # Load restaurant reviews
    reviews_df = spark.read.json('../data/reviews')

    # Randomly split data into train and test datasets
    train_df, test_df = reviews_df.randomSplit(weights=[0.75, 0.25])

    user_idx_mdl = (
        StringIndexer(inputCol="user_id", outputCol="user_idx")
        .fit(reviews_df)
    )

    business_idx_mdl = (
        StringIndexer(inputCol="business_id", outputCol="business_idx")
        .fit(reviews_df)
    )

    train_df1 = business_idx_mdl.transform(user_idx_mdl.transform(train_df))
    test_df1 = business_idx_mdl.transform(user_idx_mdl.transform(test_df))

    model = RestaurantRecommender()
    model.fit(train_df1)




if __name__ == '__main__':
    main()

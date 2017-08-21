from recommender import Recommender
import pyspark as ps
import numpy as np
import pickle
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS

spark = (
    ps.sql.SparkSession.builder
    .config('spark.driver.memory', '4g')
    .config('spark.executor.memory', '8g')
    # .master("local[8]")
    .appName("prep_website_data")
    .getOrCreate()
)


def load_ratings():
    # return spark.read.parquet('../data/ratings_ugt9_igt9')
    # return spark.read.parquet('../data/ratings_ugt1_igt1')
    # return spark.read.parquet('../data/ratings_ugt4_igt4')
    return spark.read.parquet('../data/ratings_ugt1_igt9')


def train_and_save_model_data(ratings_df):
    lambda_1 = 0.7854414047991587
    lambda_2 = 7.640976680323403
    lambda_3 = 0.0
    useALS = True
    useBias = True
    rank = 85
    regParam = 0.6762828243456599
    maxIter = 10
    nonnegative = False
    implicitPrefs = False

    estimator = Recommender(
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        useALS=useALS,
        useBias=useBias,
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        rank=rank,
        regParam=regParam,
        maxIter=maxIter,
        nonnegative=nonnegative,
        implicitPrefs=implicitPrefs
    )

    # estimator = ALS(
    #     userCol='user',
    #     itemCol='item',
    #     ratingCol='rating',
    #     rank=rank,
    #     regParam=regParam,
    #     maxIter=maxIter,
    #     nonnegative=nonnegative,
    #     implicitPrefs=implicitPrefs
    # )

    model = estimator.fit(ratings_df)

    model.itemFactors.write.parquet(
        path='../data/item_factors',
        mode='overwrite',
        compression='gzip'
    )

    model.rating_stats_df.write.parquet(
        path='../data/rating_stats',
        mode='overwrite',
        compression='gzip'
    )

    model.user_bias_df.write.parquet(
        path='../data/user_bias',
        mode='overwrite',
        compression='gzip'
    )

    model.item_bias_df.write.parquet(
        path='../data/item_bias',
        mode='overwrite',
        compression='gzip'
    )

    model.residual_stats_df.write.parquet(
        path='../data/residual_stats',
        mode='overwrite',
        compression='gzip'
    )


def main():
    ratings_df = load_ratings()

    train_and_save_model_data(ratings_df)


if __name__ == '__main__':
    main()
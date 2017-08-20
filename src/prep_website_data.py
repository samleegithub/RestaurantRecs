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
    # return spark.read.parquet('../data/ratings_ugt10_igt10')
    return spark.read.parquet('../data/ratings_ugt1_igt1')
    # return spark.read.parquet('../data/ratings_ugt4_igt4')


def train_and_save_model_data(ratings_df):
    lambda_1 = 0.4993032990785937
    lambda_2 = 0.754143704958773
    lambda_3 = 0.0
    useALS = True
    useBias = True
    rank = 81
    regParam = 0.001735836550439328
    maxIter = 9
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
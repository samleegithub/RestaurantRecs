from recommender import Recommender
import pyspark as ps
import numpy as np
import pickle
import pyspark.sql.functions as F

spark = (
    ps.sql.SparkSession.builder
    # .master("local[8]")
    .appName("prep_website_data")
    .getOrCreate()
)


def load_ratings():
    # return spark.read.parquet('../data/ratings_ugt10_igt10')
    return spark.read.parquet('../data/ratings_ugt1_igt1')


def save_discount_factor(ratings_df):
    discount_factor_df = (
        ratings_df
        .groupBy('item')
        .count()
        .select(
            F.col('item'),
            F.col('count').alias('num_ratings'),
            (1 - (1 / F.pow(F.col('count'), 0.5))).alias('discount_factor')
        )
    )

    discount_factor_df.write.parquet(
        path='../data/discount_factor',
        mode='overwrite',
        compression='gzip'
    )


def train_and_save_model_data(ratings_df):
    lambda_1 = 0.5
    lambda_2 = 0.5
    useALS = True
    useBias = False
    rank = 2
    regParam = 0.7
    maxIter = 15
    nonnegative = True
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

    model = estimator.fit(ratings_df)

    model.itemFactors.write.parquet(
        path='../data/item_factors',
        mode='overwrite',
        compression='gzip'
    )

    model.avg_rating_df.write.parquet(
        path='../data/avg_rating',
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


def main():
    ratings_df = load_ratings()
    save_discount_factor(ratings_df)
    train_and_save_model_data(ratings_df)


if __name__ == '__main__':
    main()
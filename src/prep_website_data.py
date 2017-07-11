from recommender import Recommender
import pyspark as ps
import numpy as np
import pickle

spark = (
    ps.sql.SparkSession.builder
    # .master("local[8]")
    .appName("prep_website_data")
    .getOrCreate()
)

def train_and_save_model_data():
    # Load restaurant reviews
    ratings_df = spark.read.parquet('../data/ratings_ugt10_igt10')

    lambda_1 = 7
    lambda_2 = 12
    useALS = True
    useBias = True
    rank = 76
    regParam = 0.7
    maxIter = 15
    nonnegative=True
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

    item_factors_df = model.itemFactors

    item_factors_list = []
    item_ids_list = []
    for row in model.itemFactors.collect():
        item_factors_list.append(row['features'])
        item_ids_list.append(row['id'])

    item_factors = np.array(item_factors_list)
    item_ids = np.array(item_ids_list)

    with open('../data/item_factors.pkl', 'wb') as f:
        pickle.dump(item_factors, f)

    with open('../data/item_ids.pkl', 'wb') as f:
        pickle.dump(item_ids, f)


def main():
    train_and_save_model_data()



if __name__ == '__main__':
    main()
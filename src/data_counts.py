from scrape_yelp_reviews import load_restaurant_ids
import pyspark as ps


def main():
    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("data_counts")
        .getOrCreate()
    )

    restaurants_df = spark.read.parquet('../data/restaurants')
    print('Restaurants count: {}'.format(restaurants_df.count()))

    restaurants_by_state_df = spark.read.parquet('../data/restaurants_by_state')
    print('Restaurants_by_state count: {}'.format(restaurants_by_state_df.count()))


if __name__ == '__main__':
    main()

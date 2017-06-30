from scrape_yelp_reviews import load_restaurant_ids
import pyspark as ps


def main():
    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("data_counts")
        .getOrCreate()
    )

    restaurant_ids = load_restaurant_ids(spark)
    print('Restaurants: {}'.format(len(restaurant_ids)))


if __name__ == '__main__':
    main()

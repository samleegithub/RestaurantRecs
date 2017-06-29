import json
from yelp_api import YelpAPI
from collections import Counter
import pyspark as ps


def get_restaurants():
    zipcodes_filename = '../data/zipcodes.json'
    with open(zipcodes_filename, 'r') as f:
        zipcodes_by_city = json.load(f)

    yelp_api = YelpAPI()

    restaurants = []
    for key in zipcodes_by_city:
        for zipcode in zipcodes_by_city[key]:
            print('Downloading zipcode: {0}'.format(zipcode))
            limit = 50
            page = 0
            while True:
                data = yelp_api.search_restaurants(
                    location=zipcode, radius=0, sort_by='distance',
                    limit=limit, page=page)
                page_restaurants = (data.json()['businesses'])
                num_page_restaurants = len(page_restaurants)
                restaurants += page_restaurants

                # Count number of zip codes returned
                zips = [
                    restaurant['location']['zip_code']
                    for restaurant in page_restaurants]
                print(
                    'num restaurants: {0} {1}'
                    .format(num_page_restaurants, sorted(Counter(zips).items())))

                # Check to see if there are more to request
                if num_page_restaurants < limit:
                    break
                else:
                    page = page + 1

    return restaurants


def save_restaurants(restaurants, restaurants_filename):
    with open(restaurants_filename, 'w') as f:
        for restaurant in restaurants:
            f.write(json.dumps(restaurant) + '\n')


def convert_to_parquet(spark, restaurants_filename):
    restaurants_df = spark.read.json(restaurants_filename)
    restaurants_dedup_df = restaurants_df.dropDuplicates(['id'])
    print('restaurants_df schema:')
    print(restaurants_df.printSchema())
    print('raw count        : {0}'.format(restaurants_df.count()))
    print('after dedup count: {0}'.format(restaurants_dedup_df.count()))

    restaurants_dedup_df.write.parquet(
        path='../data/restaurants',
        mode='overwrite',
        compression='gzip'
    )


def main():
    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("get_restaurant_data")
        .getOrCreate()
    )

    restaurants = get_restaurants()
    restaurants_filename = '../data/restaurants.json'
    save_restaurants(restaurants, restaurants_filename)
    convert_to_parquet(spark, restaurants_filename)



if __name__ == '__main__':
    main()

import json
from yelp_api import YelpAPI
from collections import Counter
import pyspark as ps
from pyspark.sql.functions import col


def get_restaurants(zipcodes_filename):
    '''
    Iterates through all zipcodes for each city specified in zipcodes_filename
    file. Uses Yelp's business search API to download restaurants data for each
    zipcode.

    Inputs
    ======
    zipcodes_filename       Path to file containing zipcodes grouped by city in
                            json format.

    Returns
    =======
    restaurants             List of dictionaries with each dictionary
                            containing restaurant data from Yelp API.

    '''
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
                if (data.status_code == 400 and
                        data.json()['error']['code'] == 'LOCATION_NOT_FOUND'):
                    # Zipcode is invalid. Skip.
                    print('Yelp says that zipcode {} is invalid. Skipping.'
                        .format(zipcode))
                    break

                page_restaurants = (data.json()['businesses'])
                num_page_restaurants = len(page_restaurants)
                restaurants += page_restaurants

                # Count number of zip codes returned
                zips = [
                    restaurant['location']['zip_code']
                    if restaurant['location']['zip_code'] is not None
                    else 'N/A'
                    for restaurant in page_restaurants
                ]
                zip_counts = Counter(zips)
                print(
                    'num restaurants: {0} {1}'
                    .format(num_page_restaurants, sorted(zip_counts.items()))
                )

                # Check to see if there are more to request
                if num_page_restaurants < limit:
                    break
                else:
                    page = page + 1

    return restaurants


def save_restaurants(restaurants, restaurants_filename):
    '''
    Save restaurants data to file in json format. Each line in the file
    contains a json representation of one restaurant.

    Inputs
    ======
    restaurants             List of dictionaries. Each dictionary represents
                            one restaurant.
    restaurants_filename    File path to save data to

    Outputs
    =======
    None
    '''
    with open(restaurants_filename, 'w') as f:
        for restaurant in restaurants:
            f.write(json.dumps(restaurant) + '\n')


def convert_to_parquet(spark, restaurants_filename, restaurants_parquet_path):
    '''
    Convert json restaurant data file to compressed parquet files.

    Inputs
    ======
    spark                   spark session
    restaurants_filename    name of json file to convert to to parquet

    Outputs
    =======
    None
    '''
    restaurants_df = spark.read.json(restaurants_filename)
    restaurants_dedup_df = restaurants_df.dropDuplicates(['id'])
    print('restaurants_df schema:')
    print(restaurants_df.printSchema())
    print('raw count        : {0}'.format(restaurants_df.count()))
    print('after dedup count: {0}'.format(restaurants_dedup_df.count()))

    # remove restaurants already saved if running "by state" version
    if restaurants_parquet_path[-9:] == '_by_state':
        saved_restaurant_ids = {
            row['id']
            for row in (
                spark.read.parquet('../data/restaurants')
                .select('id')
                .toLocalIterator()
            )
        }

        restaurants_dedup_df = (
            restaurants_dedup_df
            .filter(restaurants_dedup_df['id'].isin(saved_restaurant_ids) == False)
        )

    restaurants_dedup_df.write.parquet(
        path=restaurants_parquet_path,
        mode='overwrite',
        compression='gzip'
    )


def main():
    '''
    Read in zipcodes from a file and save restaurant data from Yelp's API into
    compressed spark parquet files.

    Inputs
    ======
    None

    Outputs
    =======
    None
    '''
    # Add suffix for data downloaded by state instead of the original method
    # which was by city (Seattle and San Francisco)
    # suffix = '_by_state'
    suffix = ''
    zipcodes_filename = '../data/zipcodes{}.json'.format(suffix)
    restaurants_filename = '../data/restaurants{}.json'.format(suffix)
    restaurants_parquet_path = '../data/restaurants{}'.format(suffix)

    print(zipcodes_filename)
    print(restaurants_filename)
    print(restaurants_parquet_path)

    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("get_restaurant_data")
        .getOrCreate()
    )

    restaurants = get_restaurants(zipcodes_filename)
    save_restaurants(restaurants, restaurants_filename)
    convert_to_parquet(spark, restaurants_filename, restaurants_parquet_path)


if __name__ == '__main__':
    main()

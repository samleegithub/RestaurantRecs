import json
from yelp_api import YelpAPI
from collections import Counter
import pyspark as ps


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
                page_restaurants = (data.json()['businesses'])
                num_page_restaurants = len(page_restaurants)
                restaurants += page_restaurants

                # Count number of zip codes returned
                zips = [
                    restaurant['location']['zip_code']
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


def convert_to_parquet(spark, restaurants_filename):
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

    restaurants_dedup_df.write.parquet(
        path='../data/restaurants',
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
    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("get_restaurant_data")
        .getOrCreate()
    )

    zipcodes_filename = '../data/zipcodes.json'
    restaurants = get_restaurants(zipcodes_filename)
    restaurants_filename = '../data/restaurants.json'
    save_restaurants(restaurants, restaurants_filename)
    convert_to_parquet(spark, restaurants_filename)


if __name__ == '__main__':
    main()

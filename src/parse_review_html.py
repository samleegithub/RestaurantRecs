from pymongo import MongoClient
from bs4 import BeautifulSoup
import pyspark as ps
import datetime
import json
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, ByteType


def load_html_from_db():
    '''
    Load yelp html data from MongoDB.

    Inputs
    ======
    None

    Outputs
    =======
    raw_html_data       Generator the returns (yelp_id, html_str) for each page
                        of Yelp restaurant reviews.
    '''
    client = MongoClient()
    yelp_db = client['yelp']
    reviews_table = yelp_db['reviews']

    return (
        (item['yelp_id'], item['html_str'], item['page'], item['num_pages'])
        for item in reviews_table.find()
    )


def parse_html(raw_html_data):
    '''
    Iterates through raw html data and returns a list of dictionaries with the
    following fields:

    user_id             User identifier
    product_id          Restaurant identifier
    rating              Star rating. Integer value from 1 to 5

    Input
    =====
    raw_html_data       Generator that returns (yelp_id, html_str)

    Return
    ======
    parsed_data         List of dictionaries. Each dictionary will have fields
                        specified in the method description.
    '''

    ratings = []

    for i, (restaurant_id, html_str, page, num_pages) in enumerate(raw_html_data):
        soup = BeautifulSoup(html_str, 'html.parser')
        user_passport_infos = soup.find_all('ul', class_='user-passport-info')
        reviews = soup.find_all('div', class_='review-content')

        # lengths of both user_passport_infos and reviews should always be
        # equal to 20 or less. In addition, they should be equal to each other.
        assert len(user_passport_infos) <= 20
        assert len(reviews) <= 20
        assert len(user_passport_infos) == len(reviews)

        for user_passport_info, review in zip(user_passport_infos, reviews):
            a__user_display_name = (
                user_passport_info
                .find('a', class_='user-display-name')
            )

            if a__user_display_name:
                user_id = a__user_display_name['href'].split('=')[1]
            else:
                qype_user = (
                    user_passport_info
                    .find('span', class_='ghost-qype-user')
                )

                if qype_user:
                    print('Qype user encountered. No user_id available so '
                        + 'skipping...')
                    break
                else:
                    print('ERROR!!! Unexpected issue in user scraping '
                        + 'encountered!')
                    print('restaurant_id: {}'.format(restaurant_id))
                    print('page: {}/{}'.format(page, num_pages))
                    exit()

            rating = (
                review
                .find('div', class_='rating-large')['title']
                .split()[0]
                .split('.')[0]
            )

            ratings.append(
                {
                    'user_id' : user_id,
                    'product_id' : restaurant_id,
                    'rating' : int(rating)
                }
            )

        # print status update every 1000 pages
        if i % 100 == 0:
            print('{}: Number of ratings parsed: {} Pages parsed: {}'
                .format(datetime.datetime.now(), len(ratings), i + 1))

    return ratings


def save_ratings(ratings, ratings_filename):
    '''
    Save ratings data to file in json format. Each line in the file contains a
    json representation of one rating.

    Inputs
    ======
    ratings                 List of dictionaries. Each dictionary represents
                            one rating.
    ratings_filename        File path to save data to

    Outputs
    =======
    None
    '''
    with open(ratings_filename, 'w') as f:
        for rating in ratings:
            f.write(json.dumps(rating) + '\n')


def convert_to_parquet(spark, ratings_filename):
    '''
    Convert json ratings data file to compressed parquet files.

    Inputs
    ======
    spark                   spark session
    ratings_filename        name of json file to convert to to parquet

    Outputs
    =======
    None
    '''
    ratings_df = spark.read.json(ratings_filename)
    ratings_dedup_df = ratings_df.dropDuplicates(['user_id', 'product_id'])
    print('ratings_df schema:')
    print(ratings_df.printSchema())
    print('raw count        : {0}'.format(ratings_df.count()))
    print('after dedup count: {0}'.format(ratings_dedup_df.count()))

    # encode user_id and product_id into integers
    user_idx_mdl = (
        StringIndexer(inputCol='user_id', outputCol='user_idx')
        .fit(ratings_dedup_df)
    )

    product_idx_mdl = (
        StringIndexer(inputCol='product_id', outputCol='product_idx')
        .fit(ratings_dedup_df)
    )

    # cast user_id and product_id to IntegerType
    # cast rating to ByteType. Valid values for rating are integers from 0 to 5
    ratings_df2 = (
        product_idx_mdl.transform(
            user_idx_mdl.transform(
                ratings_dedup_df
            )
        )
        .select(
            col('user_idx').cast(IntegerType()).alias('user_id'),
            col('product_idx').cast(IntegerType()).alias('product_id'),
            col('rating').cast(ByteType()).alias('rating')
        )
    )

    print('ratings_df2 schema:')
    print(ratings_df2.printSchema())

    ratings_df2.write.parquet(
        path='../data/ratings',
        mode='overwrite',
        compression='gzip'
    )

    # save user and product labels to files
    user_labels = user_idx_mdl.labels
    product_labels = product_idx_mdl.labels

    with open('../data/user_labels.txt', 'w') as f:
        for user_label in user_labels:
            f.write('{}\n'.format(user_label))

    with open('../data/product_labels.txt', 'w') as f:
        for product_label in product_labels:
            f.write('{}\n'.format(product_label))


def main():
    # print('Loading raw html data from MongoDB...')
    # raw_html_data = load_html_from_db()
    #
    # print('Parsing html into ratings...')
    # ratings = parse_html(raw_html_data)
    #
    # print('Saving to json file...')
    ratings_filename = '../data/ratings.json'
    # save_ratings(ratings, ratings_filename)

    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("parse_review_html")
        .getOrCreate()
    )

    print('Converting to Spark parquet file format...')
    convert_to_parquet(spark, ratings_filename)


if __name__ == '__main__':
    main()

from pymongo import MongoClient
from bs4 import BeautifulSoup
import pyspark as ps


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
        (item['yelp_id'], item['html_str'])
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

    for restaurant_id, html_str in raw_html_data:
        soup = BeautifulSoup(html_str, 'html.parser')
        user_passport_infos = soup.find_all('ul', class_='user-passport-info')
        reviews = soup.find_all('div', class_='review-content')

        # lengths of both user_passport_infos and reviews should always be
        # equal to 20 or less. In addition, they should be equal to each other.
        assert len(user_passport_infos) <= 20
        assert len(reviews) <= 20
        assert len(user_passport_infos) == len(reviews)

        for user_passport_info, review in zip(user_passport_infos, reviews):
            user_id = (
                user_passport_info
                .find('a', class_='user-display-name')['href']
                .split('=')[1]
            )

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

    return ratings


def save_ratings(ratings, ratings_filename):
    '''
    Save restaurants data to file in json format. Each line in the file
    contains a json representation of one restaurant.

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
    print(restaurants_df.printSchema())
    print('raw count        : {0}'.format(ratings_df.count()))
    print('after dedup count: {0}'.format(ratings_dedup_df.count()))

    ratings_dedup_df.write.parquet(
        path='../data/ratings',
        mode='overwrite',
        compression='gzip'
    )


def main():
    raw_html_data = load_html_from_db()
    ratings = parse_html(raw_html_data)
    ratings_filename = '../data/ratings.json'
    save_ratings(ratings, ratings_filename)

    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("parse_review_html")
        .getOrCreate()
    )

    convert_to_parquet(spark, ratings_filename)


if __name__ == '__main__':
    main()

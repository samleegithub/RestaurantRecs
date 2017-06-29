import requests
from bs4 import BeautifulSoup
import pyspark as ps
import numpy as np


def load_restaurant_ids(spark):
    return [
        row['id']
        for row in (
            spark.read.parquet('../data/restaurants')
            .select(['id'])
            .toLocalIterator()
        )
    ]

def get_user_agents_with_probs():
    user_agents_with_probs = np.array([
        [0.160, 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.084, 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.055, 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.040, 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/603.2.4 (KHTML, like Gecko) Version/10.1.1 Safari/603.2.4'],
        [0.024, 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'],
        [0.022, 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.021, 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.020, 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'],
        [0.017, 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.016, 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.015, 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko'],
        [0.013, 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0'],
        [0.012, 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'],
        [0.012, 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0'],
        [0.010, 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.86 Safari/537.36'],
        [0.010, 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:53.0) Gecko/20100101 Firefox/53.0']
    ])


    p = user_agents_with_probs[:,0].astype(np.float64)
    p = p / np.sum(p)
    user_agents = user_agents_with_probs[:,1]

    return p, user_agents

def scrape_reviews(yelp_id, probs, user_agents):
    user_agent = np.random.choice(user_agents, p=probs)

    base_url = 'https://www.yelp.com/biz/{0}'
    reviews_per_page = 20
    url = base_url.format(yelp_id)
    print(url, user_agent)


def main():
    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("get_restaurant_data")
        .getOrCreate()
    )

    yelp_ids = load_restaurant_ids(spark)

    probs, user_agents = get_user_agents_with_probs()

    for yelp_id in yelp_ids:
        scrape_reviews(yelp_id, probs, user_agents)



if __name__ == '__main__':
    main()

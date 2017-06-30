import requests
from bs4 import BeautifulSoup
import pyspark as ps
import numpy as np
import time
import random
from pymongo import MongoClient


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
    probs = user_agents_with_probs[:,0].astype(np.float64)
    probs = probs / np.sum(probs)
    user_agents = user_agents_with_probs[:,1]
    return probs, user_agents


def add_session_proxy(s):
    # setup luminati proxy
    username = 'lum-customer-hl_257a91b1-zone-zone1'
    password = 'ny7nlmcxto2g'
    port = 22225
    session_id = random.random()

    super_proxy_url = (
        'http://{0}-country-us-session-{1}:{2}@zproxy.luminati.io:{3}'
        .format(username, session_id, password, port)
    )

    s.proxies = {
      'http': super_proxy_url,
      'https': super_proxy_url
    }

    return s


def create_requests_session(probs, user_agents):
    # choose random user_agent string
    user_agent = np.random.choice(user_agents, p=probs)
    print('user_agent: {0}'.format(user_agent))
    s = requests.Session()
    s.headers['User-Agent'] = user_agent

    # s = add_session_proxy(s)

    return s


def get(s, url, params):
    response = s.get(url, params=params)
    if response.status_code != 200:
        print("WARNING", response.status_code, response.text)
    else:
        return response


def save_html(reviews_table, yelp_id, page, html_str):
    reviews_table.insert(
        {
            'yelp_id': yelp_id,
            'page': page,
            'html_str': html_str
        }
    )

def scrape_reviews(yelp_id, probs, user_agents, reviews_table):
    s = create_requests_session(probs, user_agents)

    base_url = 'https://www.yelp.com/biz/{0}'
    reviews_per_page = 20
    url = base_url.format(yelp_id)
    # url = 'https://lumtest.com/echo.json' # test url to confirm proxy working
    params = {}

    print('Getting page 0 ...')
    response = get(s, url, params)
    html_str = response.text
    save_html(reviews_table, yelp_id, 0, html_str)

    print('Scraping number of pages ...')
    soup = BeautifulSoup(response.text, 'html.parser')
    num_pages = int(
        soup.find('div', class_='page-of-pages')
        .getText()
        .split()[3]
    )
    print('yelp_id: {0} num_pages: {1}'.format(yelp_id, num_pages))

    for page in range(1, num_pages):
        mean = np.random.normal(10, 2)
        stdev = np.random.normal(3, 0.3)
        sleep_time = np.random.normal(mean, stdev)
        print('Sleeping for {0} seconds ...'.format(sleep_time))
        time.sleep(sleep_time)
        start = reviews_per_page * page
        params['start'] = start
        print('Getting page {0} ...'.format(page))
        response = get(s, url, params)

        html_str = response.text
        save_html(reviews_table, yelp_id, page, html_str)

    exit()



def main():
    client = MongoClient()
    yelp_db = client['yelp']
    reviews_table = yelp_db['reviews']

    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("scrape_yelp_reviews")
        .getOrCreate()
    )
    yelp_ids = load_restaurant_ids(spark)
    probs, user_agents = get_user_agents_with_probs()
    for yelp_id in yelp_ids[:1]:
        scrape_reviews(yelp_id, probs, user_agents, reviews_table)


if __name__ == '__main__':
    main()

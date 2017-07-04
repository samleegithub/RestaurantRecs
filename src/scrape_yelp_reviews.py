import requests
from bs4 import BeautifulSoup
import pyspark as ps
import numpy as np
import time
import random
from pymongo import MongoClient
import queue
import threading
import os
import json


class Worker(threading.Thread):
    def __init__(self, thread_id, q, queue_lock):
        super(Worker, self).__init__()
        self.thread_id = thread_id
        self.q = q
        self.queue_lock = queue_lock
        self._stop_event = threading.Event()

    def run(self):
        print('Thread[{}] starting...'.format(self.thread_id))
        process_data(self)
        print('Thread[{}] exiting...'.format(self.thread_id))

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def process_data(worker):
    thread_id = worker.thread_id
    q = worker.q
    queue_lock = worker.queue_lock
    while not worker.stopped():
        queue_lock.acquire()
        if not q.empty():
            data = q.get()
            queue_lock.release()
            scrape_reviews(thread_id, *data)
        else:
            queue_lock.release()
        time.sleep(1)


def load_restaurant_ids(spark, restaurants_parquet_path):
    return {
        row['id']
        for row in (
            spark.read.parquet(restaurants_parquet_path)
            .select(['id'])
            .toLocalIterator()
        )
    }


def get_saved_pages_info(reviews_table):
    restaurants = reviews_table.aggregate(
        [
            {'$group': {
                '_id': {'yelp_id': '$yelp_id', 'num_pages': '$num_pages'},
                'last_page_saved': {'$max': '$page'}
            }}
        ]
    )

    partially_saved = {}
    fully_saved = set()

    for restaurant in restaurants:
        num_pages = restaurant['_id']['num_pages']
        last_page_saved = restaurant['last_page_saved']
        yelp_id = restaurant['_id']['yelp_id']

        if last_page_saved < num_pages:
            partially_saved[yelp_id] = (last_page_saved, num_pages)
        else:
            fully_saved.add(yelp_id)

    return partially_saved, fully_saved


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
    luminati_login_filename = os.path.expanduser('~/apis/access/luminati.json')
    with open(luminati_login_filename) as f:
        luminati_login = json.load(f)

    username = luminati_login['username']
    password = luminati_login['password']
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
    # print('user_agent: {0}'.format(user_agent))
    s = requests.Session()
    s.headers['User-Agent'] = user_agent

    s = add_session_proxy(s)

    return s


def get(thread_id, s, url, params):
    response = s.get(url, params=params)

    max_retries = 10
    num_retries = 1
    # status_code 503 means that Yelp is denying access
    while response.status_code == 503 and num_retries <= max_retries:
        print('Thread[{}]: Got 503 response. Retry #{}'
            .format(thread_id, num_retries))
        # try to get a new ip address by adding a new random proxy session
        s = add_session_proxy(s)
        response = s.get(url, params=params)
        num_retries += 1

    if response.status_code != 200:
        print("WARNING", response.status_code, response.text)
    else:
        return response


def save_html(reviews_table, yelp_id, num_pages, page, html_str):
    reviews_table.insert(
        {
            'yelp_id': yelp_id,
            'num_pages': num_pages,
            'page': page,
            'html_str': html_str
        }
    )


def get_random_sleep_time():
    sleep_time = 0
    while sleep_time < 2:
        mean = np.random.normal(10, 2)
        stdev = np.random.normal(3, 0.3)
        sleep_time = np.random.normal(mean, stdev)
    return sleep_time


def scrape_reviews(thread_id, yelp_id, last_page_saved, num_pages, probs,
        user_agents, reviews_table):
    time.sleep(get_random_sleep_time())

    s = create_requests_session(probs, user_agents)

    base_url = 'https://www.yelp.com/biz/{0}'
    reviews_per_page = 20
    url = base_url.format(yelp_id)
    # url = 'https://lumtest.com/echo.json' # test url to confirm proxy working
    params = {}

    if last_page_saved is None:
        response = get(thread_id, s, url, params)
        html_str = response.text

        soup = BeautifulSoup(html_str, 'html.parser')
        tag_page_of_pages = soup.find('div', class_='page-of-pages')
        if tag_page_of_pages is not None:
            num_pages = int(tag_page_of_pages.getText().split()[3])
        else:
            print('Thread[{}]: ERROR: page-of-pages tag for {} is missing!'
                .format(thread_id, yelp_id))
            return

        print('Thread[{}]: num_pages: {} saved: {} yelp_id: {}'
            .format(thread_id, num_pages, None, yelp_id))

        save_html(reviews_table, yelp_id, num_pages, 0, html_str)
        last_page_saved = 1

    print('Thread[{}]: num_pages: {} saved: {} yelp_id: {}'
        .format(thread_id, num_pages, last_page_saved, yelp_id))

    for page in range(last_page_saved + 1, num_pages + 1):
        time.sleep(get_random_sleep_time())
        params['start'] = reviews_per_page * (page - 1)
        print('Thread[{}]: page: {}/{} yelp_id: {}'
            .format(thread_id, page, num_pages, yelp_id))
        response = get(thread_id, s, url, params)
        html_str = response.text
        save_html(reviews_table, yelp_id, num_pages, page, html_str)


def create_worker_threads(num_threads, work_queue, queue_lock):
    # Create new threads
    threads = []
    for thread_id in range(num_threads):
        thread = Worker(thread_id, work_queue, queue_lock)
        thread.start()
        threads.append(thread)
    return threads


def main():
    # Add suffix for data downloaded by state instead of the original method
    # which was by city (Seattle and San Francisco)
    # suffix = '_by_state'
    suffix = ''
    restaurants_parquet_path = '../data/restaurants{}'.format(suffix)
    table_name = 'reviews{}'.format(suffix)

    client = MongoClient()
    yelp_db = client['yelp']
    reviews_table = yelp_db[table_name]

    partially_saved, fully_saved = get_saved_pages_info(reviews_table)

    spark = (
        ps.sql.SparkSession.builder
        # .master("local[8]")
        .appName("scrape_yelp_reviews")
        .getOrCreate()
    )
    all_yelp_ids = load_restaurant_ids(spark, restaurants_parquet_path)

    yelp_ids_to_download = list(all_yelp_ids - fully_saved)

    print('Total Restaurants: {0}'.format(len(all_yelp_ids)))
    print('Total Restaurants In Progress: {0}'.format(len(partially_saved)))
    print('Total Restaurants Saved: {0}'.format(len(fully_saved)))
    print('Total Restaurants Left: {0}'.format(len(yelp_ids_to_download)))

    probs, user_agents = get_user_agents_with_probs()

    num_threads = 50
    work_queue = queue.Queue()
    queue_lock = threading.Lock()
    threads = create_worker_threads(num_threads, work_queue, queue_lock)

    # Fill the work queue
    for yelp_id in yelp_ids_to_download:
        if yelp_id in partially_saved:
            last_page_saved, num_pages = partially_saved[yelp_id]
        else:
            last_page_saved, num_pages = None, None
        data = (yelp_id, last_page_saved, num_pages, probs, user_agents,
            reviews_table)
        queue_lock.acquire()
        work_queue.put(data)
        queue_lock.release()

    print('Done filling queue')

    # Wait for work queue to empty
    while not work_queue.empty():
        pass

    print('queue is empty!')

    # Notify threads it's time to exit
    print('Notify threads it is time to exit')
    for t in threads:
        t.stop()

    # Wait for all threads to complete
    print('waiting for threads to complete')
    for t in threads:
        t.join()

    print('Exiting main thread ...')


if __name__ == '__main__':
    main()

from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
import json
import os.path
import time
from yelp_api import YelpAPI

def get_restaurants():
    pass


def main():
    client = MongoClient()
    db = client['yelp']
    tab = db['restaurants']

    yelp_api = YelpAPI()

    data = yelp_api.search_restaurants('98155')
    print(data.text)
    # tab.insert(data)


if __name__ == '__main__':
    main()

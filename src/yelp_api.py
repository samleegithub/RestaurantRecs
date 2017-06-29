import requests
import json
import os.path
import requests_interface as ri

# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
TOKEN_PATH = '/oauth2/token'
GRANT_TYPE = 'client_credentials'

class YelpAPI(object):

    def __init__(self):
        self.__token = self.__get_token()
        self.__headers = {
            'Authorization': 'Bearer {0}'.format(self.__token['access_token'])
        }

    def __get_token(self):
        yelp_access_filename = '/Users/samuellee/apis/access/yelp.json'
        token_filename = '/Users/samuellee/apis/access/yelp_token.json'
        if os.path.isfile(token_filename):
            with open(token_filename, 'r') as f:
                token = json.load(f)
        else:
            token_url = 'https://api.yelp.com/oauth2/token'
            with open(yelp_access_filename, 'r') as f:
                params = json.load(f)
            token = ri.post(url=token_url, params=params)
            with open(token_filename, 'w') as f:
                json.dump(token, f)
        return token

    def search_restaurants(self, location, radius=40000, sort_by='distance',
            limit=50, page=0):
        url = 'https://api.yelp.com/v3/businesses/search'
        offset = page * limit
        params = {
            'location': location,
            'radius': radius,
            'categories': 'restaurants',
            'sort_by': sort_by,
            'limit': limit,
            'offset': offset
        }
        return ri.get(url=url, params=params, headers=self.__headers)

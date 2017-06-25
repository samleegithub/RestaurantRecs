import requests
from bs4 import BeautifulSoup
import requests_interface as ri
import json

def get_zip_codes(city, state, pages):
    url = (
        'https://tools.usps.com/go/ZipLookupResultsAction!input.action?'
        + 'items=50&page={0}&address1=&address2=&city={1}&state={2}'
    )

    zipcodes = []
    for page in range(1, pages + 1):
        full_url = url.format(page, city, state)
        response = ri.get(full_url)
        # print(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        zipcodes += [tag.text for tag in soup.find_all('span', class_='zip')]

    return zipcodes


def main():
    cities = [
        ('Seattle', 'WA', 2),
        ('San Francisco', 'CA', 2)
    ]
    zipcodes_by_city = {}
    for city, state, pages in cities:
        zipcodes_by_city['{0}, {1}'.format(city, state)] = (
            get_zip_codes(city, state, pages)
        )

    for key in zipcodes_by_city:
        city_zipcodes = zipcodes_by_city[key]
        print('{0}:'.format(key))
        print('{0}'.format(city_zipcodes))
        print('{0} zipcodes'.format(len(city_zipcodes)))
        print()

    zipcodes_filename = '../data/zipcodes.json'
    with open(zipcodes_filename, 'w') as f:
        json.dump(zipcodes_by_city, f)


if __name__ == '__main__':
    main()

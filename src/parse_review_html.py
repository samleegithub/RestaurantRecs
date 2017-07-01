from pymongo import MongoClient
from bs4 import BeautifulSoup
import pyspark as ps


def load_html_from_db():
    client = MongoClient()
    yelp_db = client['yelp']
    reviews_table = yelp_db['reviews']

    return (
        (item['yelp_id'], item['html_str'])
        for item in reviews_table.find()
    )


def parse_html(raw_html_data):
    for restaurant_id, html_str in raw_html_data:
        soup = BeautifulSoup(html_str, 'html.parser')

        page = int(
            soup.find('div', class_='page-of-pages')
            .getText()
            .split()[1]
        )

        num_pages = int(
            soup.find('div', class_='page-of-pages')
            .getText()
            .split()[3]
        )



        soup2 = soup.find_all('div', class_='review--with-sidebar')
        for item in soup2:
            print(item)

        # ratings = soup.find_all('div', class_='rating-large')
        # print(len(ratings))
        # for rating in ratings:
        #     print(rating)
        #     print(float(rating['title'].split()[0]))


        print('{}/{}'.format(page, num_pages))
        print(len(soup2))
        exit()


def main():
    raw_html_data = load_html_from_db()
    parse_html(raw_html_data)


if __name__ == '__main__':
    main()

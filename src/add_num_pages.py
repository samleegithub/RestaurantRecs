from pymongo import MongoClient
from bs4 import BeautifulSoup


def main():
    client = MongoClient()
    yelp_db = client['yelp']
    reviews_table = yelp_db['reviews']

    # get html of first page for all saved restaurants
    yelp_first_page_htmls = reviews_table.find({
        'page': 0,
        'num_pages': {'$exists': False}
    })
    bulk = reviews_table.initialize_ordered_bulk_op()
    for item in yelp_first_page_htmls:
        yelp_id = item['yelp_id']
        html_str = item['html_str']
        soup = BeautifulSoup(html_str, 'html.parser')

        # figure out the number of pages for each restaurant
        num_pages = int(
            soup.find('div', class_='page-of-pages')
            .getText()
            .split()[3]
        )

        # add num_pages to records in db.
        # reviews_table.update_many(
        #     {'yelp_id': yelp_id},
        #     {'$set': {'num_pages': num_pages}}
        # )

        (
            bulk.find({'yelp_id': yelp_id})
                .update({'$set': {'num_pages': num_pages}})
        )
    bulk.execute()


if __name__ == '__main__':
    main()

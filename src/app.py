from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pyspark as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T

app = Flask(__name__)
PORT = 5353

with open('../data/item_factors.pkl', 'rb') as f:
    item_factors = pickle.load(f)

with open('../data/item_ids.pkl', 'rb') as f:
    item_ids = pickle.load(f)

spark = (
    ps.sql.SparkSession.builder
    # .master("local[8]")
    .appName("webapp")
    .getOrCreate()
)

# Load restaurant metadata
restaurants_df = spark.read.parquet('../data/restaurants')

# Load restaurant ids
with open('../data/product_labels.txt') as f:
    restaurant_ids = [line.strip() for line in f]

print(restaurant_ids[:10])


def find_str_in_categories(categories, keyword):
    for row in categories:
            if keyword in row['alias'].lower():
                return True
    return False


find_str_in_categories_udf = F.udf(find_str_in_categories, T.BooleanType())


@app.route('/search', methods=['POST'])
def search():
    keyword = str(request.form['keyword']).lower()
    location = str(request.form['location']).lower()
    # print(keyword, location)
    data = restaurants_df.filter(
        (
            F.lower(F.col('name')).like('%{}%'.format(keyword))
            | find_str_in_categories_udf(F.col('categories'), F.lit(keyword))
        )
        & (
            F.lower(F.col('location.city')).like('%{}%'.format(location))
            | F.lower(F.col('location.address1')).like('%{}%'.format(location))
            | F.lower(F.col('location.address2')).like('%{}%'.format(location))
            | F.lower(F.col('location.address3')).like('%{}%'.format(location))
            | F.lower(F.col('location.zip_code')).like('%{}%'.format(location))
            | F.lower(F.col('location.state')).like('%{}%'.format(location))
        )
    ).collect()
    results = {}

    for row in data:
        results[row['id']] = {
            'name': row['name'],
            'url': row['url'],
            'image_url': row['image_url'],
            'location': row['location'],
            'rating': row['rating'],
            'categories': row['categories']
        }

    restaurants_df.printSchema()

    # restaurants_df.printSchema()
    return jsonify(results)


@app.route('/recommend', methods=['POST'])
def recommend():
    json_doc = request.json

    return 'Hello world! Hoping to recommend stuff here'


@app.route('/check')
def check():
    restaurants_df.printSchema()
    return 'Hi!'


@app.route('/')
def index():
    return render_template('index.html')


def main():    
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)


if __name__ == '__main__':
    main()

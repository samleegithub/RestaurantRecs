from flask import Flask, request, render_template
import pickle
import numpy as np
import pyspark as ps

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


@app.route('/recommend', methods=['POST', 'GET'])
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

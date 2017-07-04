import logging
from pyspark.ml.recommendation import ALS
from collections import defaultdict
import pyspark.sql.functions as F
import pyspark.sql.types as T

class RestaurantRecommender(object):
    '''
    Implementation of a restaurant recomendation model.
    '''

    def __init__(self, rank=10):
        '''
        Setup logger

        Parameters
        ==========
        None

        Returns
        =======
        None
        '''
        self.logger_ = logging.getLogger('resto-reco')
        self.rank = 10


    def fit(self, ratings):
        '''
        Fit ALS model using reviews as training data.

        Parameters
        ==========
        ratings (pyspark.sql.DataFrame)
            Data used to train ALS model. Columns are 'user_id',
            'product_id', and 'rating'. Values of user_id and product_id
            must be numeric. Values of rating range from 1 to 5.

        Returns
        =======
        self
        '''
        self.logger_.info("starting fit")

        self.average_ = ratings.agg(F.avg(F.col('rating'))).first()[0]

        users = ratings.groupBy('user_id')

        self.average_rating_user_ = defaultdict(
            int,
            (
                (row['user_id'], row['avg(rating)'] - self.average_)
                for row in users.avg('rating').collect()
            )
        )

        restaurants = ratings.groupBy('product_id')

        self.average_rating_restaurant_ = defaultdict(
            int,
            (
                (row['product_id'], row['avg(rating)'] - self.average_)
                for row in restaurants.avg('rating').collect()
            )
        )

        als_model = ALS(
            userCol='user_id',
            itemCol='product_id',
            ratingCol='rating',
            nonnegative=True,
            regParam=0.1,
            rank=self.rank
        )

        self.recommender_ = als_model.fit(ratings)

        self.logger_.info("finishing fit")
        return(self)


    def transform(self, requests_df):
        '''
        Predicts the rating for requested users and restaurants.

        Parameters
        ==========
        requests_df (pyspark.sql.DataFrame)
            Data used to request predictions of ratings. Columns are 'user_id'
            and 'product_id'. Values of 'user_id' and 'product_id' must be
            numeric.

        Returns
        =======
        predictions (pyspark.sql.DataFrame)
            Predictions with 'user_id', 'product_id' and 'prediction', the
            predicted value for rating. Rating will be a floating point number.

        '''
        self.logger_.info("starting predict")
        # self.logger_.debug("request count: {}".format(requests_df.count()))

        predictions_df = self.recommender_.transform(requests_df)

        # Workaround to avoid pickle errors in UserDefinedFunction method
        average = self.average_
        average_rating_user = self.average_rating_user_
        average_rating_restaurant = self.average_rating_restaurant_

        get_baseline_rating = F.UserDefinedFunction(
            lambda user_id, product_id :
                average
                + average_rating_user[user_id]
                + average_rating_restaurant[product_id],
            T.FloatType()
        )

        predictions_df2 = (
            predictions_df
            .withColumn(
                'fillblanks',
                get_baseline_rating(
                    predictions_df['user_id'],
                    predictions_df['product_id']
                )
            )
        )

        # print(predictions_df2.printSchema())

        predictions_df3 = (
            predictions_df2
            .select(
                'user_id',
                'product_id',
                'rating',
                F.nanvl('prediction', 'fillblanks').alias('prediction')
            )
        )

        self.logger_.info("finishing predict")

        return predictions_df3

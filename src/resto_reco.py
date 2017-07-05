import logging
from pyspark.ml.recommendation import ALS
from collections import defaultdict
import pyspark.sql.functions as F
import pyspark.sql.types as T
import time

class RestaurantRecommender(object):
    '''
    Implementation of a restaurant recomendation model.
    '''

    def __init__(self, rank=1):
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
        self.rank = rank


    def fit(self, ratings, rank=None):
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
        if rank:
            self.rank = rank

        print('Starting average calc...')
        start_time = time.monotonic()
        step_start_time = time.monotonic()

        self.average_ = (
            ratings
            .groupBy()
            .avg('rating')
            .withColumnRenamed('avg(rating)', 'avg_rating')
        )
        print('Average calc done in {} seconds'
            .format(time.monotonic() - step_start_time))

        step_start_time = time.monotonic()

        self.average_rating_user_ = (
            ratings
            .groupBy('user_id')
            .avg('rating')
            .withColumnRenamed('avg(rating)', 'avg_user_rating_diff')
        )
        print('User average calc done in {} seconds.'
            .format(time.monotonic() - step_start_time))

        step_start_time = time.monotonic()

        self.average_rating_restaurant_ = (
            ratings
            .groupBy('product_id')
            .avg('rating')
            .withColumnRenamed('avg(rating)', 'avg_product_rating_diff')
        )
        print('Restaurant average calc done in {} seconds.'
            .format(time.monotonic() - step_start_time))

        als_model = ALS(
            userCol='user_id',
            itemCol='product_id',
            ratingCol='rating',
            nonnegative=True,
            regParam=0.1,
            rank=self.rank
        )

        self.recommender_ = als_model.fit(ratings)
        print('Model fit done in {} seconds.'
            .format(time.monotonic() - start_time))

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
        start_time = time.monotonic()
        step_start_time = time.monotonic()

        predictions_df = self.recommender_.transform(requests_df)

        print('ALS model transform done in {} seconds'
            .format(time.monotonic() - step_start_time))

        # Workaround to avoid pickle errors in UserDefinedFunction method
        average = self.average_
        average_rating_user = self.average_rating_user_
        average_rating_restaurant = self.average_rating_restaurant_

        step_start_time = time.monotonic()

        predictions_df2 = (
            predictions_df
            .join(average_rating_user, on='user_id')
            .fillna({'avg_user_rating_diff': 0.0})
            .join(average_rating_restaurant, on='product_id')
            .fillna({'avg_product_rating_diff': 0.0})
            .crossJoin(average)
            .withColumn(
                'fillblanks',
                F.col('avg_rating') + F.col('avg_user_rating_diff') + F.col('avg_product_rating_diff')
            )
        )

        print('fillblanks calc done in {} seconds'
            .format(time.monotonic() - step_start_time))

        step_start_time = time.monotonic()

        predictions_df3 = (
            predictions_df2
            .select(
                'user_id',
                'product_id',
                'rating',
                F.nanvl('prediction', 'fillblanks').alias('prediction')
            )
        )

        print('Nan backfill done in {} seconds'
            .format(time.monotonic() - step_start_time))

        print('Model transform done in {} seconds.'
            .format(time.monotonic() - start_time))

        self.logger_.info("finishing predict")

        return predictions_df3

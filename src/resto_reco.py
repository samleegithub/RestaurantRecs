import logging
from pyspark.ml.recommendation import ALS
from collections import defaultdict
import pyspark.sql.functions as F
import pyspark.sql.types as T

class RestaurantRecommender(object):
    '''
    Implementation of a restaurant recomendation model.
    '''

    def __init__(self):
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


    def fit(self, reviews):
        '''
        Fit ALS model using reviews as training data.

        Parameters
        ==========
        reviews (pyspark.sql.DataFrame)
            Data used to train ALS model. Columns are 'user_id',
            'business_id', and 'stars'. Values of user_id and business_id
            must be numeric. Values of stars range from 1 to 5.

        Returns
        =======
        self
        '''
        self.logger_.info("starting fit")

        self.average_ = reviews.agg(F.avg(F.col('stars'))).first()[0]

        users = reviews.groupBy('user_id')

        self.average_rating_user_ = defaultdict(
            int,
            (
                (row['user_id'], row['avg(stars)'] - self.average_)
                for row in users.avg('stars').collect()
            )
        )

        restaurants = reviews.groupBy('business_id')

        self.average_rating_restaurant_ = defaultdict(
            int,
            (
                (row['business_id'], row['avg(stars)'] - self.average_)
                for row in restaurants.avg('stars').collect()
            )
        )

        als_model = ALS(
            itemCol='business_id',
            userCol='user_id',
            ratingCol='stars',
            nonnegative=True,
            regParam=0.1,
            rank=10
        )

        self.recommender_ = als_model.fit(reviews)

        self.logger_.info("finishing fit")
        return(self)


    def transform(self, requests_df):
        '''
        Predicts the stars rating for requested users and restaurants.

        Parameters
        ==========
        requests_df (pyspark.sql.DataFrame)
            Data used to request predictions of stars ratings. Columns are
            'user_id' and 'business_id'. Values of 'user_id' and
            'business_id' must be numeric.

        Returns
        =======
        predictions (pyspark.sql.DataFrame)
            Predictions with 'user_id', 'business_id' and 'prediction', the
            predicted value for stars. Stars will be a floating point number.

        '''
        self.logger_.info("starting predict")
        # self.logger_.debug("request count: {}".format(requests_df.count()))

        predictions_df = self.recommender_.transform(requests_df)

        # Workaround to avoid pickle errors in UserDefinedFunction method
        average = self.average_
        average_rating_user = self.average_rating_user_
        average_rating_restaurant = self.average_rating_restaurant_

        get_baseline_rating = F.UserDefinedFunction(
            lambda user_id, business_id :
                average
                + average_rating_user[user_id]
                + average_rating_restaurant[business_id],
            T.FloatType()
        )

        predictions_df2 = (
            predictions_df
            .withColumn(
                'fillblanks',
                get_baseline_rating(
                    predictions_df['user_id'],
                    predictions_df['business_id']
                )
            )
        )

        # print(predictions_df2.printSchema())

        predictions_df3 = (
            predictions_df2
            .select(
                'business_id',
                'user_id',
                'stars',
                F.nanvl('prediction', 'fillblanks').alias('prediction')
            )
        )

        self.logger_.info("finishing predict")

        return predictions_df3

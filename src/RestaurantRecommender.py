import logging
from pyspark.ml.recommendation import ALS
from collections import defaultdict
from pyspark.sql.functions import col, avg, UserDefinedFunction

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
            Data used to train ALS model. Columns are 'user_idx',
            'business_idx', and 'stars'. Values of user_idx and business_idx
            must be numeric. Values of stars range from 1 to 5.

        Returns
        =======
        self
        '''
        self.logger_.debug("starting fit")

        self.average_ = reviews.agg(avg(col('stars'))).first()[0]

        users = reviews.groupBy('user_idx')

        self.average_rating_user_ = defaultdict(
            int,
            (
                (row['user_idx'], row['avg(stars)'] - self.average_)
                for row in users.avg('stars').collect()
            )
        )

        restaurants = reviews.groupBy('business_idx')

        self.average_rating_restaurant_ = defaultdict(
            int,
            (
                (row['business_idx'], row['avg(stars)'] - self.average_)
                for row in restaurants.avg('stars').collect()
            )
        )

        als_model = ALS(
            itemCol='business_idx',
            userCol='user_idx',
            ratingCol='stars',
            nonnegative=True,
            regParam=0.1,
            rank=10
        )

        self.recommender_ = als_model.fit(reviews)

        self.logger_.debug("finishing fit")
        return(self)


    def transform(self, requests):
        '''
        Predicts the stars rating for requested users and restaurants.

        Parameters
        ==========
        requests (pyspark.sql.DataFrame)
            Data used to request predictions of stars ratings. Columns are
            'user_idx' and 'business_idx'. Values of 'user_idx' and
            'business_idx' must be numeric.

        Returns
        =======
        predictions (pyspark.sql.DataFrame)
            Predictions with 'user_idx', 'business_idx' and 'prediction', the
            predicted value for stars. Stars will be a floating point number.

        '''
        self.logger_.debug("starting predict")
        self.logger_.debug("request count: {}".format(requests.shape[0]))

        predictions = self.recommender_.transform(requests_df)

        get_baseline_rating = UserDefinedFunction(
            lambda user_idx, business_idx :
                self.average_
                + self.average_rating_user_[user_idx]
                + self.average_rating_restaurant_[business_idx]
        )

        predictions2 = (
            predictions
            .withColumn(
                "fillblanks",
                get_baseline_rating(
                    predictions['user_idx'],
                    predictions['business_idx']
                )
            )
        )

        predictions3 = (
            predictions2
            .na.fill(
                value=predictions_pdf["fillblanks"],
                subset='prediction'
            )
        )

        self.logger_.debug("finishing predict")

        return predictions3

from pyspark import keyword_only
from pyspark.ml.pipeline import Estimator, Model
from pyspark.ml.param.shared import *
from pyspark.ml.recommendation import ALS, ALSModel
import pyspark.sql.functions as F
import pyspark.sql.types as T
import time


class BiasALS(ALS):
    '''
    Implementation of custom recommender model.
    '''
    useALS = Param(Params._dummy(), "useALS", "whether to use " +
                   "ALS model or only rating average and user and item biases",
                   typeConverter=TypeConverters.toBoolean)

    lambda_1 = Param(Params._dummy(), "lambda_1", "regularization parameter "
                     + "for item bias",
                     typeConverter=TypeConverters.toInt)

    lambda_2 = Param(Params._dummy(), "lambda_2", "regularization parameter "
                     + "for user bias",
                     typeConverter=TypeConverters.toInt)


    @keyword_only
    def __init__(self, useALS=True, lambda_1=25, lambda_2=10, rank=10,
                 maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10,
                 implicitPrefs=False, alpha=1.0, userCol="user",
                 itemCol="item", seed=None, ratingCol="rating",
                 nonnegative=False, checkpointInterval=10,
                 intermediateStorageLevel="MEMORY_AND_DISK",
                 finalStorageLevel="MEMORY_AND_DISK"):

        super(BiasALS, self).__init__()
        self._setDefault(useALS=True, lambda_1=25, lambda_2=10, rank=10,
                         maxIter=10, regParam=0.1, numUserBlocks=10,
                         numItemBlocks=10, implicitPrefs=False, alpha=1.0,
                         userCol="user", itemCol="item", ratingCol="rating",
                         nonnegative=False, checkpointInterval=10,
                         intermediateStorageLevel="MEMORY_AND_DISK",
                         finalStorageLevel="MEMORY_AND_DISK")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)



    @keyword_only
    def setParams(self, useALS=True, lambda_1=25, lambda_2=10, rank=10,
                  maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10,
                  implicitPrefs=False, alpha=1.0, userCol="user",
                  itemCol="item", seed=None, ratingCol="rating",
                  nonnegative=False, checkpointInterval=10,
                  intermediateStorageLevel="MEMORY_AND_DISK",
                  finalStorageLevel="MEMORY_AND_DISK"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def setUseALS(self, value):
        """
        Sets the value of :py:attr:`useALS`.
        """
        return self._set(useALS=value)


    def getUseALS(self):
        """
        Gets the value of useALS or its default value.
        """
        return self.getOrDefault(self.useALS)

    def setLambda_1(self, value):
        """
        Sets the value of :py:attr:`lambda_1`.
        """
        return self._set(lambda_1=value)

    def getLambda_1(self):
        """
        Gets the value of rank or its default value.
        """
        return self.getOrDefault(self.lambda_1)

    def setLambda_2(self, value):
        """
        Sets the value of :py:attr:`lambda_2`.
        """
        return self._set(lambda_2=value)

    def getLambda_2(self):
        """
        Gets the value of rank or its default value.
        """
        return self.getOrDefault(self.lambda_2)


    def fit(self, ratings_df):
        '''
        Fit ALS model using reviews as training data.

        Parameters
        ==========
        ratings_df      (pyspark.sql.DataFrame) Data used to train recommender
                        model. Columns are 'user', 'item', and 'rating'. Values
                        of user and item must be numeric. Values of rating
                        range from 1 to 5.

        Returns
        =======
        BiasALSModel
        '''
        avg_rating_df = (
            ratings_df
            .groupBy()
            .avg(self.getRatingCol())
            .withColumnRenamed('avg({})'.format(self.getRatingCol()),
                               'avg_rating')
        )

        item_bias_df = (
            ratings_df
            .crossJoin(avg_rating_df)
            .groupBy(self.getItemCol())
            .agg(
                F.sum(
                    F.col(self.getRatingCol()) 
                    - F.col('avg_rating')
                ).alias('sum_diffs'),
                F.count("*").alias('ct')
            )
            .withColumn(
                'item_bias',
                F.col('sum_diffs')
                / (self.getLambda_1() + F.col('ct'))
            )
            .select(
                self.getItemCol(),
                'item_bias'
            )
        )

        user_bias_df = (
            ratings_df
            .crossJoin(avg_rating_df)
            .join(item_bias_df, on=self.getItemCol())
            .groupBy(self.getUserCol())
            .agg(
                F.sum(
                    F.col(self.getRatingCol())
                    - F.col('avg_rating')
                    - F.col('item_bias')
                ).alias('sum_diffs'),
                F.count("*").alias('ct')
            )            
            .withColumn(
                'user_bias',
                F.col('sum_diffs')
                / (self.getLambda_2() + F.col('ct'))
            )
            .select(
                self.getUserCol(),
                'user_bias'
            )
        )

        if self.getUseALS():
            # print('Fit using ALS!')
            residual_df = (
                ratings_df
                .crossJoin(avg_rating_df)
                .join(user_bias_df, on=self.getUserCol())
                .join(item_bias_df, on=self.getItemCol())
                .withColumn(
                    self.getRatingCol(),
                    F.col(self.getRatingCol())
                    - F.col('avg_rating')
                    - F.col('user_bias')
                    - F.col('item_bias')
                )
                .select(
                    self.getUserCol(),
                    self.getItemCol(),
                    self.getRatingCol()
                )
            )

            als_model = super(BiasALS, self).fit(residual_df)
        else:
            # print('Fit without ALS!')
            als_model = None


        return (
            BiasALSModel(self.getUseALS(), als_model._java_obj, avg_rating_df,
                user_bias_df, item_bias_df)
        )


class BiasALSModel(ALSModel):
    def __init__(self, useALS, als_model, avg_rating_df, user_bias_df,
                 item_bias_df):
        super(BiasALSModel, self).__init__()
        self.useALS = useALS
        self.als_model = als_model
        self.avg_rating_df = avg_rating_df
        self.user_bias_df = user_bias_df
        self.item_bias_df = item_bias_df

    def _transform(self, requests_df):
        '''
        Predicts the rating for requested users and restaurants.

        Parameters
        ==========
        requests_df         (pyspark.sql.DataFrame) Data used to request 
                            predictions of ratings. Columns are 'user' and
                            'item'. Values of 'user' and 'item' must be
                            numeric.

        Returns
        =======
        predictions_df      (pyspark.sql.DataFrame) Predictions with 'user',
                            'item' and 'prediction'. Prediction will be a 
                            floating point number.

        '''
        if self.useALS:
            return (
                self.als_model.transform(requests_df)
                .crossJoin(self.avg_rating_df)
                .join(self.user_bias_df, on='user')
                .join(self.item_bias_df, on='item')
                .fillna({
                    'prediction': 0.0,
                    'user_bias': 0.0,
                    'item_bias': 0.0
                })
                .withColumn(
                    'prediction',
                    F.col('prediction')
                    + F.col('avg_rating')
                    + F.col('user_bias')
                    + F.col('item_bias')
                )
                .select(
                    'user',
                    'item',
                    'rating',
                    'prediction'
                )
            )
        else:
            return (
                requests_df
                .crossJoin(self.avg_rating_df)
                .join(self.user_bias_df, on='user')
                .join(self.item_bias_df, on='item')
                .fillna({
                    'user_bias': 0.0,
                    'item_bias': 0.0
                })
                .withColumn(
                    'prediction',
                    F.col('avg_rating')
                    + F.col('user_bias')
                    + F.col('item_bias')
                )
                .select(
                    'user',
                    'item',
                    'rating',
                    'prediction'
                )
            )

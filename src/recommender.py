from pyspark import keyword_only
from pyspark.ml.pipeline import Estimator, Model
from pyspark.ml.param.shared import *
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
import pyspark.sql.types as T
import time

class Recommender(Estimator, HasCheckpointInterval, HasMaxIter,
                  HasPredictionCol, HasRegParam, HasSeed):
    '''
    Implementation of custom recommender model.
    '''
    useALS = Param(Params._dummy(), "useALS", "whether to use " +
                          "ALS model or simple rating averages",
                          typeConverter=TypeConverters.toBoolean)

    useBias = Param(Params._dummy(), "useBias", "whether to use " +
                          "bias with ALS model",
                          typeConverter=TypeConverters.toBoolean)

    lambda_1 = Param(Params._dummy(), "lambda_1", "regularization parameter "
                     + "for item bias",
                     typeConverter=TypeConverters.toFloat)

    lambda_2 = Param(Params._dummy(), "lambda_2", "regularization parameter "
                     + "for user bias",
                     typeConverter=TypeConverters.toFloat)

    lambda_3 = Param(Params._dummy(), "lambda_3", "regularization parameter "
                     + "to prevent items with low rating count from being highly recommended",
                     typeConverter=TypeConverters.toFloat)

    rank = Param(Params._dummy(), "rank", "rank of the factorization",
                 typeConverter=TypeConverters.toInt)

    numUserBlocks = Param(Params._dummy(), "numUserBlocks", "number of user " +
                          "blocks", typeConverter=TypeConverters.toInt)

    numItemBlocks = Param(Params._dummy(), "numItemBlocks", "number of item " +
                          "blocks", typeConverter=TypeConverters.toInt)

    implicitPrefs = Param(Params._dummy(), "implicitPrefs", "whether to use " +
                          "implicit preference",
                          typeConverter=TypeConverters.toBoolean)

    alpha = Param(Params._dummy(), "alpha", "alpha for implicit preference",
                  typeConverter=TypeConverters.toFloat)

    userCol = Param(Params._dummy(), "userCol", "column name for user ids. " +
                    "Ids must be within the integer value range.",
                    typeConverter=TypeConverters.toString)

    itemCol = Param(Params._dummy(), "itemCol", "column name for item ids. " +
                    "Ids must be within the integer value range.",
                    typeConverter=TypeConverters.toString)

    ratingCol = Param(Params._dummy(), "ratingCol", "column name for ratings",
                      typeConverter=TypeConverters.toString)

    nonnegative = Param(Params._dummy(), "nonnegative",
                        "whether to use nonnegative constraint for least " +
                        "squares", typeConverter=TypeConverters.toBoolean)

    intermediateStorageLevel = Param(Params._dummy(), "intermediateStorageLevel",
                                     "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
                                     typeConverter=TypeConverters.toString)

    finalStorageLevel = Param(Params._dummy(), "finalStorageLevel",
                              "StorageLevel for ALS model factors.",
                              typeConverter=TypeConverters.toString)

    coldStartStrategy = Param(Params._dummy(), "coldStartStrategy", "strategy for dealing with " +
                              "unknown or new users/items at prediction time. This may be useful " +
                              "in cross-validation or production scenarios, for handling " +
                              "user/item ids the model has not seen in the training data. " +
                              "Supported values: 'nan', 'drop'.",
                              typeConverter=TypeConverters.toString)


    @keyword_only
    def __init__(self, useALS=True, useBias=True, lambda_1=0.5, lambda_2=0.5,
                 lambda_3=0.5, rank=10,
                 maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10,
                 implicitPrefs=False, alpha=1.0, userCol="user",
                 itemCol="item", seed=None, ratingCol="rating",
                 nonnegative=False, checkpointInterval=10,
                 intermediateStorageLevel="MEMORY_AND_DISK",
                 finalStorageLevel="MEMORY_AND_DISK"
                 # , coldStartStrategy="nan"
                 ):
        '''
        Parameters
        ==========
        None

        Returns
        =======
        None
        '''
        super(Recommender, self).__init__()
        self._setDefault(useALS=True, useBias=True, lambda_1=0.5, lambda_2=0.5,
                         lambda_3=0.5, rank=10,
                         maxIter=10, regParam=0.1, numUserBlocks=10,
                         numItemBlocks=10, implicitPrefs=False, alpha=1.0,
                         userCol="user", itemCol="item", ratingCol="rating",
                         nonnegative=False, checkpointInterval=10,
                         intermediateStorageLevel="MEMORY_AND_DISK",
                         finalStorageLevel="MEMORY_AND_DISK"
                         # , coldStartStrategy="nan"
                         )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)


    @keyword_only
    def setParams(self, useALS=True, useBias=True, lambda_1=0.5, lambda_2=0.5,
                  lambda_3=0.5, rank=10,
                  maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10,
                  implicitPrefs=False, alpha=1.0, userCol="user",
                  itemCol="item", seed=None, ratingCol="rating",
                  nonnegative=False, checkpointInterval=10,
                  intermediateStorageLevel="MEMORY_AND_DISK",
                  finalStorageLevel="MEMORY_AND_DISK"
                  # , coldStartStrategy="nan"
                  ):
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

    def setUseBias(self, value):
        """
        Sets the value of :py:attr:`useBias`.
        """
        return self._set(useBias=value)

    def getUseBias(self):
        """
        Gets the value of useBias or its default value.
        """
        return self.getOrDefault(self.useBias)

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

    def setLambda_3(self, value):
        """
        Sets the value of :py:attr:`lambda_3`.
        """
        return self._set(lambda_3=value)

    def getLambda_3(self):
        """
        Gets the value of rank or its default value.
        """
        return self.getOrDefault(self.lambda_3)

    def setRank(self, value):
        """
        Sets the value of :py:attr:`rank`.
        """
        return self._set(rank=value)

    def getRank(self):
        """
        Gets the value of rank or its default value.
        """
        return self.getOrDefault(self.rank)

    def setNumUserBlocks(self, value):
        """
        Sets the value of :py:attr:`numUserBlocks`.
        """
        return self._set(numUserBlocks=value)

    def getNumUserBlocks(self):
        """
        Gets the value of numUserBlocks or its default value.
        """
        return self.getOrDefault(self.numUserBlocks)

    def setNumItemBlocks(self, value):
        """
        Sets the value of :py:attr:`numItemBlocks`.
        """
        return self._set(numItemBlocks=value)

    def getNumItemBlocks(self):
        """
        Gets the value of numItemBlocks or its default value.
        """
        return self.getOrDefault(self.numItemBlocks)

    def setNumBlocks(self, value):
        """
        Sets both :py:attr:`numUserBlocks` and :py:attr:`numItemBlocks` to the specific value.
        """
        self._set(numUserBlocks=value)
        return self._set(numItemBlocks=value)

    def setImplicitPrefs(self, value):
        """
        Sets the value of :py:attr:`implicitPrefs`.
        """
        return self._set(implicitPrefs=value)

    def getImplicitPrefs(self):
        """
        Gets the value of implicitPrefs or its default value.
        """
        return self.getOrDefault(self.implicitPrefs)

    def setAlpha(self, value):
        """
        Sets the value of :py:attr:`alpha`.
        """
        return self._set(alpha=value)

    def getAlpha(self):
        """
        Gets the value of alpha or its default value.
        """
        return self.getOrDefault(self.alpha)

    def setUserCol(self, value):
        """
        Sets the value of :py:attr:`userCol`.
        """
        return self._set(userCol=value)

    def getUserCol(self):
        """
        Gets the value of userCol or its default value.
        """
        return self.getOrDefault(self.userCol)

    def setItemCol(self, value):
        """
        Sets the value of :py:attr:`itemCol`.
        """
        return self._set(itemCol=value)

    def getItemCol(self):
        """
        Gets the value of itemCol or its default value.
        """
        return self.getOrDefault(self.itemCol)

    def setRatingCol(self, value):
        """
        Sets the value of :py:attr:`ratingCol`.
        """
        return self._set(ratingCol=value)

    def getRatingCol(self):
        """
        Gets the value of ratingCol or its default value.
        """
        return self.getOrDefault(self.ratingCol)

    def setNonnegative(self, value):
        """
        Sets the value of :py:attr:`nonnegative`.
        """
        return self._set(nonnegative=value)

    def getNonnegative(self):
        """
        Gets the value of nonnegative or its default value.
        """
        return self.getOrDefault(self.nonnegative)

    def setIntermediateStorageLevel(self, value):
        """
        Sets the value of :py:attr:`intermediateStorageLevel`.
        """
        return self._set(intermediateStorageLevel=value)

    def getIntermediateStorageLevel(self):
        """
        Gets the value of intermediateStorageLevel or its default value.
        """
        return self.getOrDefault(self.intermediateStorageLevel)

    def setFinalStorageLevel(self, value):
        """
        Sets the value of :py:attr:`finalStorageLevel`.
        """
        return self._set(finalStorageLevel=value)

    def getFinalStorageLevel(self):
        """
        Gets the value of finalStorageLevel or its default value.
        """
        return self.getOrDefault(self.finalStorageLevel)

    # def setColdStartStrategy(self, value):
    #     """
    #     Sets the value of :py:attr:`coldStartStrategy`.
    #     """
    #     return self._set(coldStartStrategy=value)

    # def getColdStartStrategy(self):
    #     """
    #     Gets the value of coldStartStrategy or its default value.
    #     """
    #     return self.getOrDefault(self.coldStartStrategy)


    def _fit(self, ratings_df):
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
        self
        '''
        # avg_rating_df = (
        #     ratings_df
        #     .groupBy()
        #     .avg(self.getRatingCol())
        #     .withColumnRenamed('avg({})'.format(self.getRatingCol()),
        #                        'avg_rating')
        # )

        print('Fit starting!')

        start_time = time.monotonic()

        # print('ratings_df')
        # ratings_df.show()

        rating_stats_df = (
            ratings_df
            .agg(
                F.avg(self.getRatingCol()).alias('avg_rating'),
                F.stddev_samp(self.getRatingCol()).alias('stddev_rating')
            )
        )

        # print('ratings_stats_df:')
        # rating_stats_df.show()

        if not self.getUseALS():
            self.setLambda_1(0.0)
            self.setLambda_2(0.0)

        item_bias_df = (
            ratings_df
            .crossJoin(rating_stats_df)
            .withColumn(
                'diffs_item_rating',
                F.col(self.getRatingCol()) - F.col('avg_rating')
            )
            .groupBy(self.getItemCol())
            .agg(
                F.avg(F.col('diffs_item_rating')).alias('avg_diffs_item_rating'),
                F.nanvl(
                    F.stddev_samp(F.col('diffs_item_rating')),
                    F.lit(2.147483647E9)
                ).alias('stddev_diffs_item_rating'),
                F.count("*").alias('count_item_rating')
            )
            .withColumn(
                'stderr_diffs_item_rating',
                F.col('stddev_diffs_item_rating') / F.sqrt('count_item_rating')
            )
            .withColumn(
                'item_bias',
                F.col('avg_diffs_item_rating')
                / (1 + self.getLambda_1() * F.col('stderr_diffs_item_rating'))
            )
            .select(
                self.getItemCol(),
                'item_bias',
                'avg_diffs_item_rating',
                'stderr_diffs_item_rating',
                'stddev_diffs_item_rating',
                'count_item_rating'
            )
        )

        # print('item_bias_df:')
        # item_bias_df.show(5)

        # item_bias_df.printSchema()

        # print('item_bias_df NaN')
        # item_bias_df.where(F.isnan("item_bias")).show()

        user_bias_df = (
            ratings_df
            .crossJoin(rating_stats_df)
            .join(item_bias_df, on=self.getItemCol())
            .withColumn(
                'diffs_user_rating',
                F.col(self.getRatingCol()) - F.col('avg_rating') - F.col('item_bias')
            )
            .groupBy(self.getUserCol())
            .agg(
                F.avg(F.col('diffs_user_rating')).alias('avg_diffs_user_rating'),
                F.nanvl(
                    F.stddev_samp(F.col('diffs_user_rating')),
                    F.lit(2.147483647E9)
                ).alias('stddev_diffs_user_rating'),
                F.count("*").alias('count_user_rating')
            )
            .withColumn(
                'stderr_diffs_user_rating',
                F.col('stddev_diffs_user_rating') / F.sqrt('count_user_rating')
            )
            .withColumn(
                'user_bias',
                F.col('avg_diffs_user_rating')
                / (1 + self.getLambda_2() * F.col('stderr_diffs_user_rating')
                )
            )
            .select(
                self.getUserCol(),
                'user_bias',
                'avg_diffs_user_rating',
                'stderr_diffs_user_rating',
                'stddev_diffs_user_rating',
                'count_user_rating'
            )
        )

        # print('user_bias_df:')
        # user_bias_df.show(5)

        # print('user_bias_df NaN')
        # user_bias_df.where(F.isnan("user_bias")).show()

        if self.getUseALS():
            if self.getUseBias():
                residual_df = (
                    ratings_df
                    .crossJoin(rating_stats_df)
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

            else:
                residual_df = ratings_df
                # self.setColdStartStrategy('drop')

            residual_stats_df = (
                residual_df
                .agg(
                    F.avg(F.col(self.getRatingCol())).alias('avg_residual'),
                    F.stddev(F.col(self.getRatingCol())).alias('stddev_residual')
                )
            )

            # print('residual_df')
            # residual_df.show()

            # print('residual_df NaN')
            # residual_df.where(F.isnan("rating")).show()

            # print('residual_stats_df')
            # residual_stats_df.show()

            als_model = ALS(
                rank=self.getRank(),
                maxIter=self.getMaxIter(),
                regParam=self.getRegParam(),
                numUserBlocks=self.getNumUserBlocks(),
                numItemBlocks=self.getNumItemBlocks(),
                implicitPrefs=self.getImplicitPrefs(),
                alpha=self.getAlpha(),
                userCol=self.getUserCol(),
                itemCol=self.getItemCol(),
                ratingCol=self.getRatingCol(),
                nonnegative=self.getNonnegative(),
                checkpointInterval=self.getCheckpointInterval(),
                intermediateStorageLevel=self.getIntermediateStorageLevel(),
                finalStorageLevel=self.getFinalStorageLevel()
            )

            recommender = als_model.fit(residual_df)

        else:
            recommender = None
            residual_stats_df = None

        print('Fit done in {} seconds'.format(time.monotonic() - start_time))

        return(
            RecommenderModel(
                self.getUseALS(), self.getUseBias(), self.getLambda_3(),
                # self.getColdStartStrategy(),
                recommender, rating_stats_df, residual_stats_df,
                user_bias_df, item_bias_df
            )
        )


class RecommenderModel(Model):
    def __init__(self, useALS, useBias, lambda_3,
                 # coldStartStrategy,
                 recommender, rating_stats_df, residual_stats_df,
                 user_bias_df, item_bias_df):
        super(RecommenderModel, self).__init__()
        self.useALS = useALS
        self.useBias = useBias
        self.lambda_3 = lambda_3
        # self.coldStartStrategy = coldStartStrategy
        self.recommender = recommender
        self.rating_stats_df = rating_stats_df
        self.residual_stats_df = residual_stats_df
        self.user_bias_df = user_bias_df
        self.item_bias_df = item_bias_df


    @property
    def rank(self):
        """rank of the matrix factorization model"""
        return self.recommender.rank


    @property
    def userFactors(self):
        """
        a DataFrame that stores user factors in two columns: `id` and
        `features`
        """
        return self.recommender.userFactors

    @property
    def itemFactors(self):
        """
        a DataFrame that stores item factors in two columns: `id` and
        `features`
        """
        return self.recommender.itemFactors


    def recommendForAllUsers(self, numItems):
        """
        Returns top `numItems` items recommended for each user, for all users.
        :param numItems: max number of recommendations for each user
        :return: a DataFrame of (userCol, recommendations), where recommendations are
                 stored as an array of (itemCol, rating) Rows.
        """
        return self.recommender.recommendForAllUsers(numItems)


    def recommendForAllItems(self, numUsers):
        """
        Returns top `numUsers` users recommended for each item, for all items.
        :param numUsers: max number of recommendations for each item
        :return: a DataFrame of (itemCol, recommendations), where recommendations are
                 stored as an array of (userCol, rating) Rows.
        """
        return self.recommender.recommendForAllItems(numUsers)


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
        final_prediction_df (pyspark.sql.DataFrame) Predictions with 'user',
                            'item' and 'prediction'. Prediction will be a 
                            floating point number.

        '''

        print('Transform starting!')

        start_time = time.monotonic()

        if self.useALS:
            self.prediction_df = self.recommender.transform(requests_df)

            self.prediction_stats_df = (
                self.prediction_df
                .dropna(how='all', subset=['prediction'])
                .agg(
                    F.avg(F.col('prediction')).alias('avg_prediction'),
                    F.stddev_samp(F.col('prediction')).alias('stddev_prediction')
                )
            )

            # print('prediction_df')
            # self.prediction_df.show()

            # print('prediction_stats_df')
            # self.prediction_stats_df.show()

            # print('rating_stats_df')
            # self.rating_stats_df.show()

            # print('residual_stats_df')
            # self.residual_stats_df.show()

            if self.useBias:
                final_prediction_df = (
                    self.prediction_df
                    .crossJoin(self.rating_stats_df)
                    # .crossJoin(self.prediction_stats_df)
                    # .crossJoin(self.residual_stats_df)
                    .join(self.user_bias_df, on='user')
                    .join(self.item_bias_df, on='item')
                    .fillna({
                        'user_bias': 0.0,
                        'item_bias': 0.0
                    })
                    .withColumn(
                        'prediction',
                        (
                            F.coalesce(
                                F.col('prediction')
                                # - F.col('avg_prediction')
                                , F.lit(0.0)
                            )
                            # * F.col('stddev_residual')
                            # / F.col('stddev_prediction')
                            # + F.col('avg_residual')
                            + F.col('avg_rating')
                            + F.col('user_bias')
                            + F.col('item_bias')
                        )
                        # * (1 - (self.lambda_3 / F.sqrt(F.col('count_item_rating'))))
                    )
                   .select(
                        'user',
                        'item',
                        'rating',
                        'prediction'
                    )
                )

            else:
                final_prediction_df = (
                    self.prediction_df
                    .dropna(how='all', subset=['prediction'])
                    # .fillna({'prediction': F.col('avg_prediction')})
                    # .crossJoin(self.residual_stats_df)
                    # .crossJoin(self.prediction_stats_df)
                    # .withColumn(
                    #     'prediction',
                    #     (
                    #         F.col('prediction')
                    #         - F.col('avg_prediction')
                    #     )
                    #     * F.col('stddev_residual')
                    #     / F.col('stddev_prediction')
                    #     + F.col('avg_residual')
                    # )
                )
        else:
            final_prediction_df = (
                requests_df
                .crossJoin(self.rating_stats_df)
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

        print('Transform done in {} seconds'.format(time.monotonic() - start_time))

        # print('final_prediction_df')
        # final_prediction_df.show()

        return final_prediction_df

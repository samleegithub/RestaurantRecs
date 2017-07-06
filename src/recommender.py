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
    def __init__(self, rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10,
                 implicitPrefs=False, alpha=1.0, userCol="user", itemCol="item", seed=None,
                 ratingCol="rating", nonnegative=False, checkpointInterval=10,
                 intermediateStorageLevel="MEMORY_AND_DISK",
                 finalStorageLevel="MEMORY_AND_DISK", coldStartStrategy="nan"):
        '''
        Parameters
        ==========
        None

        Returns
        =======
        None
        '''
        super(Recommender, self).__init__()
        self._setDefault(rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10,
                         implicitPrefs=False, alpha=1.0, userCol="user", itemCol="item",
                         ratingCol="rating", nonnegative=False, checkpointInterval=10,
                         intermediateStorageLevel="MEMORY_AND_DISK",
                         finalStorageLevel="MEMORY_AND_DISK", coldStartStrategy="nan")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)


    @keyword_only
    def setParams(self, rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10,
                  implicitPrefs=False, alpha=1.0, userCol="user", itemCol="item", seed=None,
                  ratingCol="rating", nonnegative=False, checkpointInterval=10,
                  intermediateStorageLevel="MEMORY_AND_DISK",
                  finalStorageLevel="MEMORY_AND_DISK", coldStartStrategy="nan"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


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


    def setColdStartStrategy(self, value):
        """
        Sets the value of :py:attr:`coldStartStrategy`.
        """
        return self._set(coldStartStrategy=value)


    def getColdStartStrategy(self):
        """
        Gets the value of coldStartStrategy or its default value.
        """
        return self.getOrDefault(self.coldStartStrategy)


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
        avg_rating = (
            ratings_df
            .groupBy()
            .avg(self.getRatingCol())
            .withColumnRenamed('avg({})'.format(self.getRatingCol()),
                               'avg_rating')
        )

        avg_user_bias = (
            ratings_df
            .groupBy(self.getUserCol())
            .avg(self.getRatingCol())
            .withColumnRenamed('avg({})'.format(self.getRatingCol()),
                               'avg_user_rating')
            .crossJoin(avg_rating)
            .withColumn('avg_user_bias',
                        F.col('avg_user_rating') - F.col('avg_rating'))
            .select(
                self.getUserCol(),
                'avg_user_bias'
            )
        )

        avg_item_bias = (
            ratings_df
            .groupBy(self.getItemCol())
            .avg(self.getRatingCol())
            .withColumnRenamed('avg({})'.format(self.getRatingCol()),
                               'avg_item_rating')
            .crossJoin(avg_rating)
            .withColumn('avg_item_bias',
                        F.col('avg_item_rating') - F.col('avg_rating'))
            .select(
                self.getItemCol(),
                'avg_item_bias'
            )
        )

        residual_df = (
            ratings_df
            .crossJoin(avg_rating)
            .join(avg_user_bias, on=self.getUserCol())
            .join(avg_item_bias, on=self.getItemCol())
            .withColumn(
                'residual',
                F.col(self.getRatingCol())
                - F.col('avg_rating')
                - F.col('avg_user_bias')
                - F.col('avg_item_bias')
            )
            .select(
                self.getUserCol(),
                self.getItemCol(),
                'residual'
            )
        )

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
            ratingCol='residual',
            nonnegative=self.getNonnegative(),
            checkpointInterval=self.getCheckpointInterval(),
            intermediateStorageLevel=self.getIntermediateStorageLevel(),
            finalStorageLevel=self.getFinalStorageLevel()
        )

        recommender = als_model.fit(residual_df)


        return (
            RecommenderModel(recommender, avg_rating, avg_user_bias,
                avg_item_bias)
        )


class RecommenderModel(Model):
    def __init__(self, recommender, avg_rating, avg_user_bias, avg_item_bias):
        super(RecommenderModel, self).__init__()
        self.recommender = recommender
        self.avg_rating = avg_rating
        self.avg_user_bias = avg_user_bias
        self.avg_item_bias = avg_item_bias


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
        predictions_df      (pyspark.sql.DataFrame) Predictions with 'user',
                            'item' and 'prediction'. Prediction will be a 
                            floating point number.

        '''
        return (
            self.recommender.transform(requests_df)
            .crossJoin(self.avg_rating)
            .join(self.avg_user_bias, on='user')
            .join(self.avg_item_bias, on='item')
            .fillna({
                'prediction': 0.0,
                'avg_user_bias': 0.0,
                'avg_item_bias': 0.0
            })
            .withColumn(
                'prediction',
                F.col('prediction')
                + F.col('avg_rating')
                + F.col('avg_user_bias')
                + F.col('avg_item_bias')
            )
            .select(
                'user',
                'item',
                'rating',
                'prediction'
            )
        )




class NDCG10Evaluator(object):
    """
    Implementation of NDCG scoring method
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    def __init__(self, spark):
        self.spark = spark

    def evaluate(self, predictions_df):
        predictions_df.registerTempTable("predictions_df")
        score_df = self.spark.sql(
        '''
        select 1 - avg(p.dcg / a.idcg) as ndcg
        from (
            select
                x.user,
                sum(x.rating / log(2, 1 + x.pred_row_num)) as dcg
            from (
                select
                    user,
                    rating,
                    row_number() OVER (
                        PARTITION BY user
                        ORDER BY prediction DESC
                    ) as pred_row_num
                from predictions_df
            ) x 
            where x.pred_row_num <= 10
            group by x.user
        ) p
        join (
            select
                x.user,
                sum(x.rating / log(2, 1 + x.actual_row_num)) as idcg
            from (
                select
                    user,
                    rating,
                    row_number() OVER (
                        PARTITION BY user
                        ORDER BY rating DESC
                    ) as actual_row_num
                from predictions_df
            ) x 
            where x.actual_row_num <= 10
            group by x.user
        ) a on a.user = p.user
        '''
        )
        return score_df.head()[0]

    def isLargerBetter(self):
        return False


class NDCGEvaluator(object):
    """
    Implementation of NDCG scoring method
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    def __init__(self, spark):
        self.spark = spark

    def evaluate(self, predictions_df):
        predictions_df.registerTempTable("predictions_df")
        score_df = self.spark.sql(
        '''
        select 1 - avg(ndcg) as avg_ndcg
        from (
            select
                user,
                sum(dcg) / sum(idcg) as ndcg
            from (
                select
                    user,
                    rating / log(
                        2,
                        1 + row_number()
                        OVER (
                            PARTITION BY user
                            ORDER BY prediction DESC
                        )
                    ) as dcg,
                    rating / log(
                        2,
                        1 + row_number()
                        OVER (
                            PARTITION BY user
                            ORDER BY rating DESC
                        )
                    ) as idcg
                from predictions_df
            ) x
            group by user
        )
        '''
        )
        return score_df.head()[0]

    def isLargerBetter(self):
        return False


class TopQuantileEvaluator(object):
    """
    Look at 5% of most highly predicted restaurants for each user.
    Return the average actual rating of those restaurants.
    """
    def __init__(self, spark):
        self.spark = spark

    def evaluate(self, predictions_df):
        predictions_df.registerTempTable("predictions_df")
        score_df = self.spark.sql(
            '''
            select
                5.0 - avg(p.rating) as score
            from predictions_df p
            join (
                select
                    user,
                    percentile_approx(prediction, 0.95) as 95_percentile
                from predictions_df
                group by user
            ) x on p.user = x.user and p.prediction >= x.95_percentile
            '''
        )
        return score_df.head()[0]

    def isLargerBetter(self):
        return False
import pyspark.sql.functions as F

def print_ratings_counts(ratings_df, label):
    print('[{}] Num total ratings: {}'
        .format(label, ratings_df.count()))
    print('[{}] Num users: {}'
        .format(label, ratings_df.groupBy('user').count().count()))
    print('[{}] Num restaurants: {}'
        .format(label, ratings_df.groupBy('item').count().count()))
    print('[{}] Avg num ratings per user: {}'
        .format(label, ratings_df.groupBy('user').count().agg(F.avg('count')).head()[0]))
    print('[{}] Avg num ratings per restaurant: {}'
        .format(label, ratings_df.groupBy('item').count().agg(F.avg('count')).head()[0]))


def print_avg_predictions(predictions_df, label):
    result_row = (
        predictions_df
        .agg(
            F.avg('rating').alias('avg_rating'),
            F.stddev('rating').alias('stddev_rating'),
            F.avg('prediction').alias('avg_prediction'),
            F.stddev('prediction').alias('stddev_prediction')
        ).head()
    )
    print('[{} Prediction] Rating Avg: {} Stddev: {}'
        .format(label, result_row[0], result_row[1]))
    print('[{} Prediction] Prediction Avg: {} Stddev: {}'
        .format(label, result_row[2], result_row[3]))

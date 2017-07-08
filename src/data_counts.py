from scrape_yelp_reviews import load_restaurant_ids
import pyspark as ps
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def load_data(spark):
    restaurants_df = spark.read.parquet('../data/restaurants')
    ratings_df = spark.read.parquet('../data/ratings')
    return restaurants_df, ratings_df


def get_filtered_dfs(restaurants_df, ratings_df):
    user_ratings_count_df = ratings_df.groupBy('user').count()

    user_filtered_ratings_dfs = [
        (
            ratings_df
            .join(user_ratings_count_df, on='user')
            .where('count > {}'.format(i+1))
            .select('user', 'item', 'rating')
        )
        for i in range(10)
    ]

    item_ratings_count_df = ratings_df.groupBy('item').count()

    item_filtered_ratings_dfs = [
        (
            ratings_df
            .join(item_ratings_count_df, on='item')
            .where('count > {}'.format(i+1))
            .select('user', 'item', 'rating')
        )
        for i in range(10)
    ]

    return (
        user_ratings_count_df, item_ratings_count_df,
        user_filtered_ratings_dfs, item_filtered_ratings_dfs
    )


def print_counts(
    restaurants_df, ratings_df,
    user_ratings_count_df, item_ratings_count_df,
    user_filtered_ratings_dfs, item_filtered_ratings_dfs
):
    print('Restaurants count: {}'.format(restaurants_df.count()))

    # restaurants_by_state_df = spark.read.parquet('../data/restaurants_by_state')
    # print('Restaurants_by_state count: {}'.format(restaurants_by_state_df.count()))
    
    total_ratings_count = ratings_df.count()
    print('Ratings count: {}'.format(total_ratings_count))

    total_users_count = user_ratings_count_df.count()
    print('Num users: {}'.format(total_users_count))

    total_items_count = item_ratings_count_df.count()
    print('Num restaurants (from ratings data): {}'.format(total_items_count))

    # user counts
    user_counts = []
    cumulative_user_counts = []
    cumulative_user_ratings_counts = [0]
    for i in range(10):
        user_counts.append(
            user_ratings_count_df.where('count = {}'.format(i+1)).count())
        cumulative_user_counts.append(sum(user_counts))
        cumulative_user_ratings_counts.append(
            cumulative_user_ratings_counts[-1] + user_counts[i] * (i+1)
        )
        print('Num users with {} ratings: {}'
            .format(i+1, user_counts[i]))

    cumulative_user_ratings_counts = cumulative_user_ratings_counts[1:]

    # create datasets with "low rating count" users filtered out
    user_more_than_counts = []
    for i in range(10):
        user_more_than_counts.append(
            user_filtered_ratings_dfs[i].groupBy('user').count().count()
        )
        
        print('Num users with more than {} ratings: {} (should equal {})'
            .format(i+1,
                    user_more_than_counts[i],
                    total_users_count - cumulative_user_counts[i]
        ))

    for i in range(10):
        print('Num ratings of users with more than {} ratings: {} (should equal {})'
            .format(i+1,
                    user_filtered_ratings_dfs[i].count(),
                    total_ratings_count - cumulative_user_ratings_counts[i]
        ))

    # item counts
    item_counts = []
    cumulative_item_counts = []
    cumulative_item_ratings_counts = [0]
    for i in range(10):
        item_counts.append(
            item_ratings_count_df.where('count = {}'.format(i+1)).count())
        cumulative_item_counts.append(sum(item_counts))
        cumulative_item_ratings_counts.append(
            cumulative_item_ratings_counts[-1] + item_counts[i] * (i+1)
        )
        print('Num items with {} ratings: {}'
            .format(i+1, item_counts[i]))

    cumulative_item_ratings_counts = cumulative_item_ratings_counts[1:]

    item_more_than_counts = []
    for i in range(10):
        item_more_than_counts.append(
            item_filtered_ratings_dfs[i].groupBy('item').count().count()
        )
        
        print('Num items with more than {} ratings: {} (should equal {})'
            .format(i+1,
                    item_more_than_counts[i],
                    total_items_count - cumulative_item_counts[i]
        ))

    for i in range(10):
        print('Num ratings of items with more than {} ratings: {} (should equal {})'
            .format(i+1,
                    item_filtered_ratings_dfs[i].count(),
                    total_ratings_count - cumulative_item_ratings_counts[i]
        ))



def save_filtered_ratings(user_filtered_ratings_dfs, item_filtered_ratings_dfs):
    for i, user_filtered_ratings_df in enumerate(user_filtered_ratings_dfs):
        for j, item_filtered_ratings_df in enumerate(item_filtered_ratings_dfs):
            intersection_df = (
                user_filtered_ratings_df
                .join(item_filtered_ratings_df, on=(['user', 'item', 'rating']))
            )
            
            print('num ratings of users > {} ratings and items > {} ratings: {}'
                .format(i+1,
                        j+1,
                        intersection_df.count()
            ))

            intersection_df.write.parquet(
                path='../data/ratings_ugt{}_igt{}'.format(i+1, j+1),
                mode='overwrite',
                compression='gzip'
            )


def main():
    spark = (
        ps.sql.SparkSession.builder
        .master("local[8]")
        .appName("data_counts")
        .getOrCreate()
    )

    restaurants_df, ratings_df = load_data(spark)

    (
        user_ratings_count_df, item_ratings_count_df,
        user_filtered_ratings_dfs, item_filtered_ratings_dfs
    ) = get_filtered_dfs(restaurants_df, ratings_df)

    print_counts(
        restaurants_df, ratings_df,
        user_ratings_count_df, item_ratings_count_df,
        user_filtered_ratings_dfs, item_filtered_ratings_dfs
    )
    
    save_filtered_ratings(user_filtered_ratings_dfs, item_filtered_ratings_dfs)


    # user_rating_counts = [row['count'] for row in user_ratings_count_df.collect()]

    # fig, ax = plt.subplots(figsize=(15, 9))
    # ax.hist(user_rating_counts, log=True, bins=500)
    # ax.set_title('Histogram: User Rating Counts')
    # ax.set_xlabel('User Rating Counts')
    # ax.set_ylabel('Frequency (Log Scale)')
    # plt.tight_layout()
    # plt.show()



if __name__ == '__main__':
    main()

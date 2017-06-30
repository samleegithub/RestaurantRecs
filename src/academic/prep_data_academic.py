import pandas as pd
import numpy as np
from pyspark.sql.functions import array_contains, col
from pyspark.sql.types import IntegerType, ByteType
from pyspark.ml.feature import StringIndexer
import pyspark as ps

SPARK = (
    ps.sql.SparkSession.builder
    .master("local[8]")
    .appName("prep_data")
    .getOrCreate()
)

def fix_typos(restaurant_df):
    fix_typos_city = {
        'Central City Village' : 'Central City',
        'Gelndale' : 'Glendale',
        'Glenndale' : 'Glendale',
        'Laveen Village' : 'Laveen',
        'MESA' : 'Mesa',
        'Mesa AZ' : 'Mesa',
        'Pheonix' : 'Phoenix',
        'Pheonix AZ' : 'Phoenix',
        'Scottdale' : 'Scottsdale',
        'Stuttgart-Vaihingen' : 'Stuttgart',
        'City of Edinburgh' : 'Edinburgh',
        'Edimbourg' : 'Edinburgh',
        'Concord Mills' : 'Concord',
        'Harrisbug' : 'Harrisburg',
        'Mattews' : 'Matthews',
        'Mint  Hill' : 'Mint Hill',
        'Las  Vegas' : 'Las Vegas',
        'LasVegas' : 'Las Vegas',
        'N Las Vegas' : 'North Las Vegas',
        'N. Las Vegas' : 'North Las Vegas',
        'Nellis AFB' : 'Nellis Air Force Base',
        'Nellis Afb' : 'Nellis Air Force Base',
        'las vegas' : 'Las Vegas',
        'Bainbridge Township' : 'Chagrin Falls',
        'Bedford Hts.' : 'Bedford Heights',
        'Brookpark' : 'Brook Park',
        'Concord Twp' : 'Mentor',
        'Cuyahoga Fls' : 'Cuyahoga Falls',
        'Medina Township' : 'Medina',
        'Mentor On the' : 'Mentor-on-the-Lake',
        'Mentor On the Lake' : 'Mentor-on-the-Lake',
        'N. Olmsted' : 'North Olmsted',
        'North Olmstead' : 'North Olmsted',
        'WICKLIFFE' : 'Wickliffe',
        'Warrensvile Heights' : 'Warrensville Heights',
        'Warrensville Hts.' : 'Warrensville Heights',
        'columbia station' : 'Columbia Station',
        'AGINCOURT' : 'Agincourt',
        'E Gwillimbury' : 'East Gwillimbury',
        'Missisauga' : 'Mississauga',
        'Mississuaga' : 'Mississauga',
        'NORTH YORK' : 'North York',
        'Richmond Hil' : 'Richmond Hill',
        'Scaroborough' : 'Scarborough',
        'Scarobrough' : 'Scarborough',
        'TORONTO' : 'Toronto',
        'Thornhil' : 'Thornhill',
        'Vaughn' : 'Vaughan',
        'Bellvue' : 'Bellevue',
        'East Mc Keesport' : 'East McKeesport',
        'Elizabeth Township' : 'Elizabeth',
        'Mc Donald' : 'McDonald',
        'Mc Murray' : 'McMurray',
        'Moon Township' : 'Moon',
        'Moon Twp' : 'Moon',
        'Moon Twp.' : 'Moon',
        'Mt. Lebanon' : 'Mount Lebanon',
        'Robinson Township' : 'Robinson',
        'Robinson Twp.' : 'Robinson',
        'South Park Township' : 'South Park',
        'Stowe Township' : 'Stowe',
        'Upper St Clair' : 'Upper Saint Clair',
        'Upper St. Clair' : 'Upper Saint Clair',
        'Chatauguay' : 'Châteauguay',
        'Chateauguay' : 'Châteauguay',
        'Cote Saint-Luc' : 'Côte-Saint-Luc',
        'Cote-Saint-Luc' : 'Côte-Saint-Luc',
        'Dollard-Des Ormeaux' : 'Dollard-Des-Ormeaux',
        'Dollard-des-Ormeaux' : 'Dollard-Des-Ormeaux',
        "L'assomption" : "L'Assomption",
        "L'ile-Perrot" : "L'Île-Perrot",
        "L'Île-Perrôt" : "L'Île-Perrot",
        'La Salle' : 'LaSalle',
        'Montreal' : 'Montréal',
        'Montreal-Nord' : 'Montréal-Nord',
        'Montreal-Ouest' : 'Montréal-Ouest',
        'Montreal-West' : 'Montréal-Ouest',
        'Montéal' : 'Montréal',
        'Saint Laurent' : 'Saint-Laurent',
        'Saint Leonard' : 'Saint-Léonard',
        'Saint-Bruno' : 'Saint-Bruno-de-Montarville',
        'Saint-Jean-Sur-Richelieu' : 'Saint-Jean-sur-Richelieu',
        'Saint-Jerome' : 'Saint-Jérôme',
        'Saint-Leonard' : 'Saint-Léonard',
        'Saint-Marc-Sur-Richelieu' : 'Saint-Marc-sur-Richelieu',
        'Saint-Sauveur-des-Monts' : 'Saint-Sauveur',
        'Sainte-Adele' : 'Sainte-Adèle',
        'Sainte-Anne-De-Bellevue' : 'Sainte-Anne-de-Bellevue',
        'Sainte-Therese' : 'Sainte-Thérèse',
        'Sainte-Therese-de-Blainville' : 'Sainte-Thérèse',
        'Sainte-thérèse' : 'Sainte-Thérèse',
        'Salaberry-De-Valleyfield' : 'Salaberry-de-Valleyfield',
        'St Leonard' : 'Saint-Léonard',
        'St-Benoît de Mirabel' : 'Mirabel',
        'St-Jerome' : 'Saint-Jérôme',
        'St-Laurent' : 'Saint-Laurent',
        'St-Leonard' : 'Saint-Léonard',
        'Ste-Therese-de-Blainville' : 'Sainte-Thérèse',
        'Fort  Mill' : 'Fort Mill',
        'Ft. Mill' : 'Fort Mill',
        'De Forest' : 'DeForest',
        'Mc Farland' : 'McFarland',
    }
    restaurant_df2 = restaurant_df.na.replace(fix_typos_city, 'city')

    fix_typos_state = {
        'KHL' : 'MLN',
        'PKN' : 'EDH'
    }
    restaurant_df3 = restaurant_df2.na.replace(fix_typos_state, 'state')

    return restaurant_df3


def prep_restaurant_data():
    business_df = SPARK.read.json(
        '../data/yelp_dataset_challenge_round9/yelp_academic_dataset_business.json'
    )

    # print(business_df.printSchema())

    restaurant_df = (
        business_df
        .filter(array_contains('categories', 'Restaurants'))
        .persist()
    )

    print(restaurant_df.printSchema())

    restaurant_df2 = fix_typos(restaurant_df)

    restaurant_df2.write.parquet(
        path='../data/restaurants_academic',
        mode='overwrite',
        compression='gzip'
    )

    return restaurant_df2


def prep_review_data(restaurant_df):
    # cast stars to ByteType
    # valid values for stars are integers from 0 to 5
    review_df = (
        SPARK.read.json(
            path='../data/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json'
        )
        .select(
            'user_id',
            'business_id',
            col('stars').cast(ByteType()).alias('stars')
        )
    )

    # print(review_df.printSchema())

    restaurant_ids_df = restaurant_df.select('business_id')

    restaurant_review_df = review_df.join(
        other=restaurant_ids_df,
        on='business_id',
        how='inner'
    )

    user_idx_mdl = (
        StringIndexer(inputCol='user_id', outputCol='user_idx')
        .fit(restaurant_review_df)
    )

    business_idx_mdl = (
        StringIndexer(inputCol='business_id', outputCol='business_idx')
        .fit(restaurant_review_df)
    )

    # cast business_id and user_id to IntegerType
    # max business_id = 48484, max user_id = 721778
    restaurant_review_df2 = (
        business_idx_mdl.transform(
            user_idx_mdl.transform(
                restaurant_review_df
            )
        )
        .select(
            col('business_idx').cast(IntegerType()).alias('business_id'),
            col('user_idx').cast(IntegerType()).alias('user_id'),
            'stars'
        )
    )

    restaurant_review_df2.write.parquet(
        path='../data/reviews_academic',
        mode='overwrite',
        compression='gzip'
    )


def main():
    restaurant_df = prep_restaurant_data()
    prep_review_data(restaurant_df)


if __name__ == '__main__':
    main()

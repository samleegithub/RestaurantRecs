from recommender import Recommender
from model_evaluators import NDCG10Evaluator, NDCGEvaluator, TopQuantileEvaluator
from ratings_helper_functions import print_ratings_counts, print_avg_predictions
import numpy as np
import pyspark as ps
import time
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F
from hyperopt import hp
import hyperopt
from functools import partial
from pprint import pprint


spark = (
    ps.sql.SparkSession.builder
    .config('spark.executor.memory', '4g')
    # .master("local[8]")
    .appName("eval_model")
    .getOrCreate()
)

# Load restaurant reviews
# ratings_df = spark.read.parquet('../data/ratings')
ratings_df = spark.read.parquet('../data/ratings_ugt1_igt1')
# ratings_df = spark.read.parquet('../data/ratings_ugt5_igt5')
# ratings_df = spark.read.parquet('../data/ratings_ugt10_igt10')

# print(ratings_df.printSchema())
print_ratings_counts(ratings_df, 'Total')

train_df, test_df = ratings_df.randomSplit(weights=[0.5, 0.5])

print_ratings_counts(train_df, 'Train')
print_ratings_counts(test_df, 'Test')


def uniform_int(name, lower, upper):
    # `quniform` returns:
    # round(uniform(low, high) / q) * q
    return hp.quniform(name, lower, upper, q=1)


def loguniform_int(name, lower, upper):
    # Do not forget to make a logarithm for the
    # lower and upper bounds.
    return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)


def setup_parameter_space():
    parameter_space = {
        'rank': uniform_int('rank', 1, 250),
        'regParam': hp.loguniform('regParam', np.log(0.0001), np.log(2)),
        'lambda_1': hp.uniform('lambda_1', 0, 10),
        'lambda_2': hp.uniform('lambda_2', 0, 10),
        'maxIter': uniform_int('maxIter', 1, 15)
    }

    return parameter_space


def eval_model(parameters):
    print("Parameters:")
    pprint(parameters)
    print()

    rank = int(parameters['rank'])
    regParam = parameters['regParam']
    lambda_1 = parameters['lambda_1']
    lambda_2 = parameters['lambda_2']
    maxIter = int(parameters['maxIter'])

    eval_name = 'RMSE'
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")  

    estimator = Recommender(
        useALS=True,
        useBias=True,
        rank=rank,
        regParam=regParam,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=0,
        maxIter=maxIter,
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        nonnegative=False
    )

    model = estimator.fit(train_df)

    predictions_df = model.transform(test_df)

    score = evaluator.evaluate(predictions_df)

    print('score: {}\n'.format(score))

    return score


def optimize(parameter_space):
    # Object stores all information about each trial.
    # Also, it stores information about the best trial.
    trials = hyperopt.Trials()

    tpe = partial(
        hyperopt.tpe.suggest,
        # Sample 1000 candidate and select candidate that
        # has highest Expected Improvement (EI)
        n_EI_candidates=1000,
        # Use 20% of best observations to estimate next
        # set of parameters
        gamma=0.2,
        # First 20 trials are going to be random
        n_startup_jobs=20,
    )

    hyperopt.fmin(
        eval_model,
        trials=trials,
        space=parameter_space,
        # Set up TPE for hyperparameter optimization
        algo=tpe,
        # Maximum number of iterations. Basically it runs at
        # most 200 times before selecting the best one.
        max_evals=200,
    )

    return trials


def main():
    parameter_space = setup_parameter_space()

    trials = optimize(parameter_space)

    print("trials.best_trial:")
    pprint(trials.best_trial)
    print()


if __name__ == '__main__':
    main()
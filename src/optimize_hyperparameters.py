from recommender import Recommender
from model_evaluators import NDCG10Evaluator, NDCGEvaluator, TopQuantileEvaluator
from ratings_helper_functions import print_ratings_counts, print_avg_predictions
import numpy as np
import pyspark as ps
import time
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F
import hyperopt
from functools import partial
from pprint import pprint
import pickle
from pathlib import Path
import signal


spark = (
    ps.sql.SparkSession.builder
    .config('spark.driver.memory', '4g')
    .config('spark.executor.memory', '8g')
    # .master("local[8]")
    .appName("optimize_hyperparameters")
    .getOrCreate()
)

# Load restaurant reviews
# ratings_filename = '../data/ratings'
# ratings_filename = '../data/ratings_ugt1_igt1'
# ratings_filename = '../data/ratings_ugt5_igt5'
# ratings_filename = '../data/ratings_ugt10_igt10'
ratings_filename = '../data/ratings_ugt9_igt9'
suffix = '_unadj'
model_filename = '{}{}.hyperopt'.format(ratings_filename, suffix)

print('Ratings filename: {}'.format(ratings_filename))
print('hyperopt filename: {}'.format(model_filename))

ratings_df = spark.read.parquet(ratings_filename)

# print(ratings_df.printSchema())
print_ratings_counts(ratings_df, 'Total')

train_df, test_df = ratings_df.randomSplit(weights=[0.5, 0.5])

print_ratings_counts(train_df, 'Train')
print_ratings_counts(test_df, 'Test')

trials = hyperopt.Trials()


def uniform_int(name, lower, upper):
    # `quniform` returns:
    # round(uniform(low, high) / q) * q
    return hyperopt.hp.quniform(name, lower, upper, q=1)


def loguniform_int(name, lower, upper):
    # Do not forget to make a logarithm for the
    # lower and upper bounds.
    return hyperopt.hp.qloguniform(name, np.log(lower), np.log(upper), q=1)


def setup_parameter_space():
    parameter_space = {
        'rank': uniform_int('rank', 1, 250),
        'regParam': hyperopt.hp.uniform('regParam', 0.001, 10),
        'lambda_1': hyperopt.hp.uniform('lambda_1', 0, 10),
        'lambda_2': hyperopt.hp.uniform('lambda_2', 0, 10),
        # 'maxIter': uniform_int('maxIter', 1, 15)
    }

    return parameter_space


def score_model(estimator, eval_name, evaluator, baseline=False):
    if baseline:
        eval_name = 'Baseline {}'.format(eval_name)

    model = estimator.fit(train_df)

    train_predictions_df = model.transform(train_df)
    test_predictions_df = model.transform(test_df)

    # print('Train score starting!')
    start_time = time.monotonic()
    train_score = evaluator.evaluate(train_predictions_df)
    print('Train score done in {} seconds'.format(time.monotonic() - start_time))

    # print('Test score starting!')
    start_time = time.monotonic()
    test_score = evaluator.evaluate(test_predictions_df)
    print('Train score done in {} seconds'.format(time.monotonic() - start_time))

    print('=========================================')
    print('{} score on Train: {}'.format(eval_name, train_score))
    print('{} score on Test: {}'.format(eval_name, test_score))
    print('=========================================')
    print('')

    return train_score, test_score


def eval_model(parameters):
    print("Parameters:")
    pprint(parameters)
    print()

    rank = int(parameters['rank'])
    regParam = parameters['regParam']
    lambda_1 = parameters['lambda_1']
    lambda_2 = parameters['lambda_2']
    # maxIter = int(parameters['maxIter'])

    estimator = Recommender(
        useALS=True,
        useBias=True,
        rank=rank,
        regParam=regParam,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=0.0,
        # maxIter=maxIter,
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        nonnegative=False
    )

    # eval_name = 'RMSE'
    # evaluator = RegressionEvaluator(
    #     metricName="rmse", labelCol="rating", predictionCol="prediction")  

    eval_name = 'NDCG10'
    evaluator = NDCG10Evaluator(spark)

    train_score, test_score = score_model(estimator, eval_name, evaluator)

    return {'loss': test_score, 'status': hyperopt.STATUS_OK}


def optimize(parameter_space):
    global trials

    # how many additional trials to do after loading saved trials.
    # 1 = save after one iteration
    trials_step = 1

    # initial max trials. put something small so we don't have to wait before
    # saving
    max_trials = 1

    model_path = Path(model_filename)

    if model_path.is_file():
        # Load an already saved trials object, and increase the max
        with open(model_filename, 'rb') as f:
            trials = pickle.load(f)

        max_trials = len(trials.trials) + trials_step

    algo_tpe = partial(
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

    best = hyperopt.fmin(
        eval_model,
        trials=trials,
        space=parameter_space,
        # Set up TPE for hyperparameter optimization
        algo=algo_tpe,
        # Maximum number of iterations.
        max_evals=max_trials,
    )

    print('Best: {}'.format(best))


def save_trials():
    # save the trials object
    with open(model_filename, "wb") as f:
        pickle.dump(trials, f)


def get_baseline_score():
    estimator = Recommender(
        useALS=False,
        useBias=True,
        rank=10,
        regParam=0.1,
        lambda_1=0.0,
        lambda_2=0.0,
        lambda_3=0.0,
        # maxIter=maxIter,
        userCol='user',
        itemCol='item',
        ratingCol='rating',
        nonnegative=False
    )

    eval_name = 'NDCG10'
    evaluator = NDCG10Evaluator(spark)

    train_score, test_score = score_model(estimator, eval_name, evaluator, baseline=True)


def main():
    get_baseline_score()

    parameter_space = setup_parameter_space()

    while True:
        optimize(parameter_space)
        save_trials()


if __name__ == '__main__':
    main()
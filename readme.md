Restaurant Recs
===============
http://restaurantrecs.com/


Idea
----
Get recommendations based on your ratings of other restaurants


Why not use Yelp?
-----------------
* Browsing and reading reviews is time-consuming
* Limited focus on only top-rated restaurants
* No personalized recommendations


Data Understanding
------------------
| Data Feature | Details |
| ------------ | --- | 
| Cities       | Seattle, San Francisco |
| Restaurants  | 5,664 |
| Users        | 510,240 |
| Ratings      | 1,672,655 |


Data Preparation
----------------
USPS.com
* Scrape all zip codes for Seattle and San Francisco

Yelp API
* Download restaurants for each zip code

Yelp.com
* Scrape all review pages for each restaurant


Model
-----
rating = avg rating + restaurant bias + user bias + residual

Non-negative Matrix Factorization trained on residual
* Implemented using PySpark ALS (Alternating Least Squares)


Model Tuning
------------
Included only users and restaurants with > 10 ratings

Tuned hyperparameters using cross validated grid search
* Rank = 76
* Regularization Parameter = 0.7


Model Evaluation
----------------
Normalized Discounted Cumulative Gain
* https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
* Scoring metric that compares against ideal ordering
* Top 10 for each user
* Perfect score is 1.0

Scores of optimized model

| Data Set | NDCG@10 Score |
| -------- | ------------- |
| Train    | 0.9366 |
| Test     | 0.9321 |


Deployment
----------
http://restaurantrecs.com

* Running on Amazon AWS EC2 instance
* Implemented algorithm to address “Cold Start” problem
  * Dot product residuals of user ratings with restaurant latent features matrix
    to get best estimate of user's latent features.


Future Steps
------------
* Allow users to login and save ratings
* Schedule nightly job to retrain model with new user ratings
* Add all restaurants for Washington, California, Oregon and Nevada


Contact Info
------------
Sam Lee

http://linkedin.com/in/sam-lee-data

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:38:01 2025

@author: ahackett
"""
'''
The task is to essentially determine whether the mean regulation energy is to be positive
or negative over the period 17:00 to 18:00 on 2021-02-16 17:00:00+00:00 to 2021-02-16 17:00:00+00:00
based on date ranging up to 2021-02-16 16:54:00+00:00



Let us start with a straightforward brute force approach, and then use this as a point of comparison with an ML approach. For a regression like this, it seems logical to make use of a random forest approach. One concern I have is that regarding the energy market in general, its very possible that the regulation energy is correlated across a day-to-day basis. For example, there are peak energy usage times during the day, and probably much less at night. However, the market is probably already very efficient and so it might not actually be the case.

I think it would be reasonable to make a brute force computation of the pdf across all time, and predict the mean regulation energy from there, then also take the means across just the time period 17:00-18:00 for each day instead. That might at least suggest whether there is a correlation here. Then I shall build a random forest regression model over either the entire dataset, or the time period 17:00-18:00, depending on the results of the initial computation. Random forest hyperparams will probably have to be set very liberally, since there is not that much time.

'''

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import sklearn as sk
import pandas as pd

ts = pd.read_csv('/Users/ahackett/Downloads/data.csv')

ts['timestamp'] = pd.to_datetime(ts['utc'])
ts['val'] = pd.to_numeric(ts['val']) # don't want to find out in an hour that this is wrong...

#Filter for 17:00 - 18:00 UTC

ts_hour = ts[ts['timestamp'].dt.hour == 17]

#Sum these
hour_tot = ts_hour.groupby(ts_hour['timestamp'].dt.date)['val'].sum()

#Also sum all the totals too
ts_all_hour = ts[ts['timestamp'].dt.hour == any]
all_tot = ts_all_hour.groupby(ts_all_hour['timestamp'].dt.date)['val'].sum()

#Compute these simple probabilities
sim_p_pos_hour_specific = (hour_tot > 0).mean()
sim_p_neg_hour_specific = 1 - sim_p_pos_hour_specific

#And just across the entire dataset
sim_p_pos_hour = (all_tot > 0).mean()
sim_p_neg_hour = 1 - sim_p_pos_hour

#Implement payoff matrix
#For the hour specific case
exp_buy_specific = sim_p_pos_hour_specific * (90 - 30) + sim_p_neg_hour_specific * (0 - 30)
exp_sell_specific = sim_p_pos_hour_specific * (30 - 90) + sim_p_neg_hour_specific * (30 - 0)
exp_nothing_specific = 0.

#For the all-time case
exp_buy_all = sim_p_pos_hour * (90 - 30) + sim_p_neg_hour * (0 - 30)
exp_sell_all = sim_p_pos_hour * (30 - 90) + sim_p_neg_hour * (30 - 0)
exp_nothing_all = 0.

#And make our descision and expected profits
decision_specific = 'buy' if exp_buy_specific > max(exp_sell_specific, exp_nothing_specific) else \
           'sell' if exp_sell_specific > max(exp_buy_specific, exp_nothing_specific) else \
           'do nothing'
'''
print(f"Probability of positive regulation energy: {sim_p_pos_hour_specific:.2%}")
print(f"Probability of negative regulation energy: {sim_p_neg_hour_specific:.2%}")
print(f"Expected profit (buy): {exp_buy_specific:.2f} EUR")
print(f"Expected profit (sell): {exp_sell_specific:.2f} EUR")
print(f"Decision: {decision_specific}")
'''

decision_all = 'buy' if exp_buy_all > max(exp_sell_all, exp_nothing_specific) else \
           'sell' if exp_sell_all > max(exp_buy_all, exp_nothing_specific) else \
           'do nothing'
'''
print(f"Probability of positive regulation energy: {sim_p_pos_hour:.2%}")
print(f"Probability of negative regulation energy: {sim_p_neg_hour:.2%}")
print(f"Expected profit (buy): {exp_buy_all:.2f} EUR")
print(f"Expected profit (sell): {exp_sell_all:.2f} EUR")
print(f"Decision: {decision_all}")
'''
#Okay the summing all the totals thing has Nan'd on me and I don't think this
#Is a sufficent solution, so I'm going to have to go on ahead, summing the specific totals
'''
That is, considering the hour between 17:00 and 18:00 on each day gives a 50.73% chance of
positive regulation energy, and a 49.27% chance of negative, So the expected profit from
buying is around 15.5 euros, and from selling -15.5, so the descision would be to buy.

This model is way to simplistic to actually make a call like this, so I'm going to set up a random
forest regression, again, over the hour in question.
'''



'''
Beginning the ML section

Actually, this doesn’t really need to be a random forest regression, I don’t care about how positive or negative the regulation energy is, so I just need a binary classifer, positive or negative.
'''

'''
I also need this script to run fairly fast, so I'm making a judgment call,
I'll make the predictor work for today, as in feb 16th 2021.
I'm almost sure this will take too long to actually run, so I shall see about restricting the 
training set to just the 15 minutes before 17:00 or something to this effect
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
Let's just make this part totally self contained to avoid headaches
'''
ts2 = pd.read_csv('/Users/ahackett/Downloads/data.csv')

ts2['timestamp'] = pd.to_datetime(ts2['utc'])
ts2['val'] = pd.to_numeric(ts2['val'])

#As before, filter for 17:00-18:00

ts2_hour = ts2[ts2['timestamp'].dt.hour == 17]
#Create the hour totals as before, and label them
hourly_totals = ts2_hour.groupby(ts2_hour['timestamp'].dt.date)['val'].sum().reset_index()
hourly_totals['label'] = (hourly_totals['val'] > 0).astype(int)
#1 for total positive, 0 for total negative

#keep features to a minumum, just day, week, and month
#In reality would want to label national holidays, by season and weather, geopolitics etc

hourly_totals['day_of_week'] = pd.to_datetime(hourly_totals['timestamp']).dt.dayofweek
hourly_totals['day_of_month'] = pd.to_datetime(hourly_totals['timestamp']).dt.day
hourly_totals['month'] = pd.to_datetime(hourly_totals['timestamp']).dt.month

#Label the features with x, and the target with y
x = hourly_totals[['day_of_week', 'day_of_month', 'month']]
y = hourly_totals['label']

'''
Now we need to split data into test and train set to ensure that the model isn't
just "learning" to emulate the dataset and that it can actually make a prediction
I do not know if a year of minute by minute is actually a lot of data in this context or not
It seems like a lot from an astronomy perspective, but we are usually data poor, so I will 
train on 80% of the data and test on 20% – could probably very well afford to train on more and test on less, but I shall chalk this call up to lack of experience in the sector!
Actually, there are over half a million minutes in a year right, so probably safe to drop to 1%
Should I shuffle the data? Usually you would always want to, but this is a time series were in principle we only make calls into the future, we probably should avoid shuffling this time so that the trends of the future aren't
being back-filled into the past, but about this I am genuinely a little confused. 
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.02, 
                                                    random_state = 2)# Well, it is *second foundation* so seemed as good a pick as any!

'''
Now train the classifer – we will need to find the optimal number of trees, ideally,
but initially lets get an accuracy and an answer with a fairly large number of trees, and 
then tune later if I have time
'''

model = RandomForestClassifier(n_estimators=10000, random_state=2)
model.fit(x_train, y_train)


'''
Benifit of using a classifer as opposed to regression where possible: much easier to evaluate accuracy!
'''

y_pred = model.predict(x_test)
y_prob = model.predict(x_test) 
expected_buy_profit_test = y_prob * (90 - 30) + (1 - y_prob) * (0 - 30)
actual_buy_test_profit = (y_test > 0) * (90 - 30) + (1 - y_test > 0) * (0 - 30)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

print(f"Model Metric: {accuracy_score(y_test, y_pred) / 0.33:.2f}")

#I won't have time to do it, but based on this model accuracy, I would look at some hyperparameter tuning before getting to making the prediction for "today"
'''
With 10k trees, I got an accuracy of 0.55 or so, which is nowhere near good enough to make the call
I think in theory you cannot worsen a random forest by adding trees
'''
# Predict for Feb 16, 2021
date = pd.to_datetime('2021-02-16')
features = pd.DataFrame({
    'day_of_week': [date.dayofweek],
    'day_of_month': [date.day],
    'month': [date.month]
})
prediction = model.predict(features)[0]
prob_positive = model.predict_proba(features)[0][1]

print(f"Predicted label (1=positive, 0=negative): {prediction}")
print(f"Probability of positive regulation energy: {prob_positive:.2f}")

#Now, just like before, compute the expected profit

# Expected profit calculations
expected_buy = prob_positive * (90 - 30) + (1 - prob_positive) * (0 - 30)
expected_sell = prob_positive * (29.99 - 90) + (1 - prob_positive) * (29.99 - 0)
expected_nothing = 0

# Decision
decision = 'buy' if expected_buy > max(expected_sell, expected_nothing) else \
           'sell' if expected_sell > max(expected_buy, expected_nothing) else \
           'do nothing'

print(f"Expected profit (buy): {expected_buy:.2f} EUR")
print(f"Expected profit (sell): {expected_sell:.2f} EUR")
print(f"Decision: {decision}")



'''
Okay, having done this, with a model accuracy of just 33%  the probability of positive regulation energy is 
36%, so the decision is to buy with an expected profit of 2.33 euros.
Practically speaking, this model accuracy is too poor – if this is all that I had at the descision point 
I could probably not justify making the call to buy on the basis of this model, I would most likely elect to do nothing, to at least avoid the potential loss
'''


'''
I think that this model and this approach are still theoretically sound, ideally I would need to consider more features, but honestly, with some hyperparam tuning I think I could get the accuracy high enough, say 95%, to consider using the decision it suggests.

I will sketch a simple Bayesian search to tune the hyperparameters

'''

from skopt import BayesSearchCV
param_space = {
    'n_estimators': (1000, 20000),
    'max_depth': (10, 50),
    'min_samples_split': (2, 30),
    'min_samples_leaf': (1, 10),
    'bootstrap': (True, False)
    
}
bayes_search = BayesSearchCV(RandomForestClassifier(), param_space, n_iter=32, cv=5, n_jobs=-1)
bayes_search.fit(x_train, y_train)

'''
How a Bayesian search works is that we map hyperparams to a probability of getting a specific score (accuracy) on the function (the model output itself) as P(score|hyperparam)

This involves a surrogate function, say, a Gaussian, or another random forest regressor for example from which the next set of values of the hyperparameters are chosen. They are chosen with a selection function, for example, an expected improvement function. 

Rather than detailing all the rest here, I will say that you could just use a simple grid based or random search across the hyperparameters which will be slower (well, depending on your choice of surrogate function and selection functions) but which is well, much simpler conceptually.

The hyperparmas I might want to optimize are the number of trees, the max depth of each tree, where shorter trees are faster, and where deeper trees can sample a more complicated/obscured pattern, but can also just overfit
Minimum samples to split a node in a tree, where higher values help stop overfitting but make you more prone to miss a patter, and bootstrapping, that determines whether taking a sample initiates a replacement or not.

'''

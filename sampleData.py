import pandas as pd

dataAll = pd.read_csv('ml-latest/ratings.csv')
print('The full dataset has {} data points'.format(len(dataAll)))
dataAll = dataAll.sample(frac=0.05, random_state=3222)
print('We have {} data points'.format(len(dataAll)))
dataAll.to_csv('data-medium/ratings_{}.csv'.format(len(dataAll), index=False))

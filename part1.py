import pickle

import pandas as pd
import numpy as np

from surprise import NMF, KNNWithMeans
from surprise.model_selection import GridSearchCV
from surprise import Dataset, Reader, evaluate
from surprise.accuracy import rmse
from surprise.accuracy import mae

from sklearn import model_selection

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

RESULTS_PATH = 'results/'


def getTrainTestData():
    ratingsDF = pd.read_csv(dataFolder + '/ratings.csv')
    ratingsDF = ratingsDF[['userId', 'movieId', 'rating']]
    ratingsTrainDF, ratingsTestDF = model_selection.train_test_split(ratingsDF, test_size=0.25, random_state=324)
    return ratingsTrainDF, ratingsTestDF


def transformToDataset(ratingsTrainDF):
    reader = Reader(rating_scale=(1, 5))
    ratingsTrainDataset = Dataset.load_from_df(ratingsTrainDF, reader)
    return ratingsTrainDataset


def runModel(ratings, model, paramGrid):
    crossValidationSplits = 3
    gridSearch = GridSearchCV(
        model,
        paramGrid,
        measures=['rmse', 'mae'],
        cv=crossValidationSplits,
        joblib_verbose=2
    )

    gridSearch.fit(ratings)

    return gridSearch


def runParameterTuning(ratingsTrainDataset):
    # NMF MODEL _______________________________________________________________________________________________________
    modelNMF = NMF
    paramGridNMF = {
        'n_epochs': [10, 25, 50],
        'n_factors': [5, 15, 30],
        'reg_pu': [0.1, 0.3, 0.5],
        'reg_qi': [0.1, 0.3, 0.5],
        'biased': [True, False],
    }
    # paramGridNMF = {
    #     # 'n_epochs': [10],
    #     'biased': [True, False],
    # }
    print('NMF')
    bestParamsNMF = {}
    for param in paramGridNMF.keys():
        print('Training {}'.format(param))
        paramGrid = {
            param: paramGridNMF[param],
            'random_state': [876],
        }
        gridSearchNMF = runModel(ratingsTrainDataset, modelNMF, paramGrid)
        saveGridSearchResults(gridSearchNMF, 'NMF', param)
        bestParamsNMF[param] = gridSearchNMF.best_params['rmse'][param]

    # KNN MODEL _______________________________________________________________________________________________________
    modelKNN = KNNWithMeans
    paramGridKNN = {
        'k': [5, 10, 20, 40, 80, 100],
        'min_k': [1, 3, 5],
    }
    # paramGridKNN = {
    #     'k': [5, 10],
    # }
    print('KNN')
    bestParamsKNN = {}
    for param in paramGridKNN.keys():
        print('Training {}'.format(param))
        paramGrid = {
            param: paramGridKNN[param],
            'random_state': [876],
            'verbose': [False],
        }
        gridSearchKNN = runModel(ratingsTrainDataset, modelKNN, paramGrid)
        saveGridSearchResults(gridSearchKNN, 'KNN', param)
        bestParamsKNN[param] = gridSearchKNN.best_params['rmse'][param]

    return bestParamsNMF, bestParamsKNN


def saveGridSearchResults(gridSearch, modelName, param):
    results = {
        'bestParams': gridSearch.best_params['rmse'],
        'bestRMSE': gridSearch.best_score['rmse'],
        'bestMAE': gridSearch.best_score['mae'],
        'cv_results': gridSearch.cv_results,
    }

    fileName = 'gridSearchResults{}_{}.p'.format(modelName, param)
    with open(fileName, 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)


def bestParamsToParamGrid(bestParams):
    paramGrid = {}
    for key, value in bestParams.items():
        paramGrid[key] = [value]
    return paramGrid


def runDataSizeTuning(ratingsTrainDF, bestParamsNMF, bestParamsKNN):
    scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    results = {
        'NMF': [],
        'KNN': [],
    }
    for scale in scales:
        ratingsTrainSampledDF = ratingsTrainDF.sample(frac=scale, random_state=939)
        ratingsTrainSampledDataset = transformToDataset(ratingsTrainSampledDF)

        numRatings = ratingsTrainSampledDF.shape[0]
        print('Running data size tuning for {} ratings'.format(numRatings))

        modelNMF = NMF
        paramGridNMF = bestParamsToParamGrid(bestParamsNMF)
        paramGridNMF['random_state'] = [2343]
        gridSearch = runModel(ratingsTrainSampledDataset, modelNMF, paramGridNMF)

        rmse = gridSearch.best_score['rmse']
        mae = gridSearch.best_score['mae']
        numRatings = ratingsTrainSampledDF.shape[0]
        time = gridSearch.cv_results['mean_fit_time'][0]

        results['NMF'].append({
            'rmse': rmse,
            'mae': mae,
            'numRatings': numRatings,
            'scale': scale,
            'time': time,
        })
        modelKNN = KNNWithMeans
        paramGridKNN = bestParamsToParamGrid(bestParamsKNN)
        paramGridKNN['random_state'] = [2343]
        paramGridKNN['verbose'] = [False]
        gridSearch = runModel(ratingsTrainSampledDataset, modelKNN, paramGridKNN)

        rmse = gridSearch.best_score['rmse']
        mae = gridSearch.best_score['mae']
        time = gridSearch.cv_results['mean_fit_time'][0]

        results['KNN'].append({
            'rmse': rmse,
            'mae': mae,
            'numRatings': numRatings,
            'scale': scale,
            'time': time,
        })

    saveDataSizeTuningResults(results)


def saveDataSizeTuningResults(results):
    for modelName in results.keys():
        modelResults = results[modelName]
        fileName = '{}dataSizeResults{}.p'.format(RESULTS_PATH, modelName)
        with open(fileName, 'wb') as fp:
            pickle.dump(modelResults, fp, protocol=pickle.HIGHEST_PROTOCOL)


def saveFinalResult(modelType, rsme, mae):
    result = {
        "modelType": modelType,
        "rsme": rsme,
        "mae": mae,
    }
    fileName = '{}finalResults{}.p'.format(RESULTS_PATH, modelType)
    with open(fileName, 'wb') as fp:
        pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)


def trainFinalModels(ratingsTrainDataset, ratingsTest, bestParamsNMF, bestParamsKNN):
    ratingsTrainTrainset = ratingsTrainDataset.build_full_trainset()

    modelNMF = NMF(**bestParamsNMF)
    modelNMF.fit(ratingsTrainTrainset)
    saveModel(modelNMF, 'NMF')

    predictions = modelNMF.test(ratingsTest)
    rmseValue = rmse(predictions)
    maeValue = mae(predictions)
    saveFinalResult('NMF', rmseValue, maeValue)

    modelKNN = KNNWithMeans(**bestParamsKNN)
    modelKNN.fit(ratingsTrainTrainset)
    saveModel(modelKNN, 'KNN')

    predictions = modelKNN.test(ratingsTest)
    rmseValue = rmse(predictions)
    maeValue = mae(predictions)
    saveFinalResult('KNN', rmseValue, maeValue)


def saveModel(modelNMF, modelType):
    filename = '{}model{}.sav'.format(RESULTS_PATH, modelType)
    pickle.dump(modelNMF, open(filename, 'wb'))


dataFolder = 'ml-latest'
# dataFolder = 'ml-latest-small'

ratingsTrainDF, ratingsTestDF = getTrainTestData()
# ratingsTrainDF = ratingsTrainDF.sample(frac=0.2, random_state=3213)
ratingsTrainDataset = transformToDataset(ratingsTrainDF)
ratingsTest = np.asarray(ratingsTestDF)
print('Train data length: {}'.format(ratingsTrainDataset.df.shape[0]))
print('Test data length: {}'.format(len(ratingsTest)))

print('')
print('Run parameter tuning')
bestParamsNMF, bestParamsKNN = runParameterTuning(ratingsTrainDataset)

print('')
print('Run Size Tuning')
runDataSizeTuning(ratingsTrainDF, bestParamsNMF, bestParamsKNN)

print('')
print('Train final models')
trainFinalModels(ratingsTrainDataset, ratingsTest, bestParamsNMF, bestParamsKNN)

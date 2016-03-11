# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:35:54 2016

@author: wpasfield
"""

import pandas as pd

url1 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/KenPom.csv'
url2 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/RegularSeasonDetailedResults.csv'
url3 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/Teams.csv'
url4 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/TourneyCompactResults.csv'
url5 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/TourneyDetailedResults.csv'
url6 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/Team_Lookup.csv'
url7 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/RegularSeasonCompactResults.csv'
url8 = 'https://raw.github.com/WesleyPasfield/March_Madness/master/Stage_1.csv'
kp = pd.read_csv(url1)
regSeason = pd.read_csv(url2)
teams = pd.read_csv(url3)
tourneyResults = pd.read_csv(url4)
tourneyDetail = pd.read_csv(url5)
teamLookup = pd.read_csv(url6)
regSeasonComp = pd.read_csv(url7)
stage1 = pd.read_csv(url8)

allDetail = pd.concat([regSeason, tourneyDetail])

## Drop headers that exist in kp CSV file

kp = kp.dropna(subset = ['Pyth_NCSOS'])

## Replace seeds from team name

kp['Team'] = kp['Team'].str.replace(' 16', '')
kp['Team'] = kp['Team'].str.replace(' 15', '')
kp['Team'] = kp['Team'].str.replace(' 14', '')
kp['Team'] = kp['Team'].str.replace(' 13', '')
kp['Team'] = kp['Team'].str.replace(' 12', '')
kp['Team'] = kp['Team'].str.replace(' 11', '')
kp['Team'] = kp['Team'].str.replace(' 10', '')
kp['Team'] = kp['Team'].str.replace(' 9', '')
kp['Team'] = kp['Team'].str.replace(' 8', '')
kp['Team'] = kp['Team'].str.replace(' 7', '')
kp['Team'] = kp['Team'].str.replace(' 6', '')
kp['Team'] = kp['Team'].str.replace(' 5', '')
kp['Team'] = kp['Team'].str.replace(' 4', '')
kp['Team'] = kp['Team'].str.replace(' 3', '')
kp['Team'] = kp['Team'].str.replace(' 2', '')
kp['Team'] = kp['Team'].str.replace(' 1', '')

## Change column names for team lookup, then merge to match KP names & Kaggle Names

teamLookup.columns = ['Team', 'Team_Name']
kp2 = kp.merge(teamLookup, on = 'Team', how = 'left')

## Drop Null cases (Winston Salem St. , irrelevant), & Merge with Teams to get team_ids

kp2['nullCheck'] = pd.isnull(kp2['Team_Name'])
kp2= kp2[kp2.nullCheck != True]
kp3 = kp2.merge(teams, on = 'Team_Name', how = 'left')

## Convert variables to correct format & drop irrelevant variables

kp3[['Pyth','AdjO','AdjD','AdjT','Luck']] = kp3[['Pyth','AdjO','AdjD','AdjT','Luck']].astype(float)
kp3.drop(['Rank','Team','Conf','nullCheck','Team_Name'],inplace=True,axis=1,errors='ignore')

## Add in total adjusted team efficiency (adjO - adjD) & create variable for Team_Year

kp3['AdjS'] = kp3['AdjO'] - kp3['AdjD']
kp3['Team_Year'] = kp3.Year.map(str) + "_" + kp3.Team_Id.map(str)
kp3 = kp3[kp3['Year'] != 2002]

## Drop Unnecessary Variables

kp3.drop(['Year', 'Team_Id'], inplace=True, axis=1, errors='ignore')

## Create z score for all variables

cols = list(kp3.columns)
cols.remove('Team_Year')

for col in cols:
    col_zscore = col + '_zscore'
    kp3[col_zscore] = (kp3[col] - kp3[col].mean())/kp3[col].std(ddof=0)
    
## Pull out only z score tranformed variables
kpFin = kp3[kp3.columns[10:21]]
kpFin.info()

## Get Team_Year combinations, then create df for all game winners & losers

allDetail['W_Team_Year'] = allDetail.Season.map(str) + '_' + allDetail.Wteam.map(str)
allDetail['L_Team_Year'] = allDetail.Season.map(str) + '_' + allDetail.Lteam.map(str)
regSeasonFin= allDetail[['W_Team_Year', 'L_Team_Year', 'possW', 'possL', 'possDiff']]

## Create KP file that just has adjO, adjD, & AdjT for merging

kp4 = kp3[['Team_Year', 'AdjO','AdjD','AdjS']]

## Merge kp3 with regSeasonFin

regSeasonFin2 = regSeasonFin.merge(kp4, left_on = 'W_Team_Year', right_on = 'Team_Year')
regSeasonFin3 = regSeasonFin2.merge(kp4, left_on = 'L_Team_Year', right_on = 'Team_Year')
regSeasonFin3.columns = ['W_Team_Year', 'L_Team_Year', 'possW', 'possL', 
                         'possDiff', 'W_Team_Year2','W_AdjO','W_AdjD','W_AdjS',
                        'L_Team_Year2', 'L_AdjO', 'L_AdjD', 'L_AdjS']

regSeasonFin4 = regSeasonFin3
regSeasonFin4['W_expO'] = (regSeasonFin3['W_AdjO'] + regSeasonFin3['L_AdjD']) / 2
regSeasonFin4['W_expD'] = (regSeasonFin3['W_AdjD'] + regSeasonFin3['L_AdjO']) / 2
regSeasonFin4['L_expO'] = (regSeasonFin3['L_AdjO'] + regSeasonFin3['W_AdjD']) / 2
regSeasonFin4['L_expD'] = (regSeasonFin3['L_AdjD'] + regSeasonFin3['W_AdjO']) / 2
regSeasonFin4['expF'] = regSeasonFin3['W_AdjS'] - regSeasonFin3['L_AdjS']
regSeasonFin4['W_ActO_ExpO'] = regSeasonFin4['possW'] - regSeasonFin4['W_expO']
regSeasonFin4['W_ActD_ExpD'] = regSeasonFin4['possL'] - regSeasonFin4['W_expD']
regSeasonFin4['L_ActO_ExpO'] = regSeasonFin4['possL'] - regSeasonFin4['L_expO']
regSeasonFin4['L_ActD_ExpD'] = regSeasonFin4['possW'] - regSeasonFin4['L_expD']
regSeasonFin4['ActF-ExpF'] = regSeasonFin4['possDiff'] - regSeasonFin4['expF']
regSeasonFin4['W_ActO_ExpO_Abs'] = abs(regSeasonFin4['possW'] - regSeasonFin4['W_expO'])
regSeasonFin4['W_ActD_ExpD_Abs'] = abs(regSeasonFin4['possL'] - regSeasonFin4['W_expD'])
regSeasonFin4['L_ActO_ExpO_Abs'] = abs(regSeasonFin4['possL'] - regSeasonFin4['L_expO'])
regSeasonFin4['L_ActD_ExpD_Abs'] = abs(regSeasonFin4['possW'] - regSeasonFin4['L_expD'])
regSeasonFin4['ActF-ExpF_Abs'] = abs(regSeasonFin4['possDiff'] - regSeasonFin4['expF'])

regSeasLoss = regSeasonFin4[['L_Team_Year','L_ActO_ExpO', 'L_ActO_ExpO_Abs', 
                             'L_ActD_ExpD','L_ActD_ExpD_Abs','ActF-ExpF','ActF-ExpF_Abs']]
                                                            
regSeasWin = regSeasonFin4[['W_Team_Year','W_ActO_ExpO', 'W_ActO_ExpO_Abs', 'W_ActD_ExpD',
                             'W_ActD_ExpD_Abs', 'ActF-ExpF', 'ActF-ExpF_Abs']]

regSeasLoss.columns = ['Team_Year', 'AdjO_Vol', 'AdjO_Vol_Abs', 'AdjD_Vol', 'AdjD_Vol_Abs', 'AdjS_Vol', 'AdjS_Vol_Abs']
regSeasWin.columns = ['Team_Year', 'AdjO_Vol', 'AdjO_Vol_Abs', 'AdjD_Vol', 'AdjD_Vol_Abs', 'AdjS_Vol', 'AdjS_Vol_Abs']

regSeasConcat = pd.concat([regSeasLoss, regSeasWin])
regSeasGroupAverage = regSeasConcat.groupby(['Team_Year']).mean()
regSeasGroupAverage['Team_Year'] = regSeasGroupAverage.index
regSeasGroupAverage = regSeasGroupAverage[['Team_Year', 'AdjO_Vol_Abs', 'AdjD_Vol_Abs', 'AdjS_Vol_Abs']]
regSeasGroupAverage.columns = ['Team_Year', 'AdjO_Vol_Avg', 'AdjD_Vol_Avg', 'AdjS_Vol_Avg']

regSeasGroupMax = regSeasConcat.groupby(['Team_Year']).max()
regSeasGroupMax['Team_Year'] = regSeasGroupMax.index
regSeasGroupMax = regSeasGroupMax[['Team_Year', 'AdjO_Vol' ,'AdjD_Vol', 'AdjS_Vol']]
regSeasGroupMax.columns = ['Team_Year', 'AdjO_Vol_Max' ,'AdjD_Vol_Max', 'AdjS_Vol_Max']

regSeasGroupMin = regSeasConcat.groupby(['Team_Year']).min()
regSeasGroupMin['Team_Year'] = regSeasGroupMin.index
regSeasGroupMin = regSeasGroupMin[['Team_Year', 'AdjO_Vol' ,'AdjD_Vol', 'AdjS_Vol']]
regSeasGroupMin.columns = ['Team_Year', 'AdjO_Vol_Min' ,'AdjD_Vol_Min', 'AdjS_Vol_Min']

regSeasGroupMaxMin = regSeasGroupMax.merge(regSeasGroupMin, on = 'Team_Year')
regSeasGroupMaxMin['AdjO_Vol_Range'] = regSeasGroupMaxMin['AdjO_Vol_Max'] - regSeasGroupMaxMin['AdjO_Vol_Min']
regSeasGroupMaxMin['AdjD_Vol_Range'] = regSeasGroupMaxMin['AdjD_Vol_Max'] - regSeasGroupMaxMin['AdjD_Vol_Min']
regSeasGroupMaxMin['AdjS_Vol_Range'] = regSeasGroupMaxMin['AdjS_Vol_Max'] - regSeasGroupMaxMin['AdjS_Vol_Min']
regSeasGroupFin = regSeasGroupMaxMin.merge(regSeasGroupAverage, on = 'Team_Year')
regSeasGroupFin.index = regSeasGroupFin['Team_Year']
regSeasGroupFin.head(n =10)

## Create z score for all variables

cols = list(regSeasGroupFin.columns)
cols.remove('Team_Year')

for col in cols:
    col_zscore = col + '_zscore'
    regSeasGroupFin[col_zscore] = (regSeasGroupFin[col] - regSeasGroupFin[col].mean())/regSeasGroupFin[col].std(ddof=0)
    
regSeasMerge = regSeasGroupFin[['Team_Year', 'AdjO_Vol_Max_zscore', 'AdjD_Vol_Max_zscore',
                               'AdjS_Vol_Max_zscore', 'AdjO_Vol_Min_zscore', 'AdjD_Vol_Min_zscore',
                               'AdjS_Vol_Min_zscore', 'AdjO_Vol_Range_zscore', 'AdjD_Vol_Range_zscore',
                               'AdjS_Vol_Range_zscore', 'AdjO_Vol_Avg_zscore', 'AdjD_Vol_Avg_zscore',
                               'AdjS_Vol_Avg_zscore']]


kpMerge = kpFin.merge(regSeasMerge, on = 'Team_Year', how = 'left')
kpLogData = kp3.merge(regSeasGroupFin, on = 'Team_Year', how = 'left')
kpMerge.columns

## Take exp of all variables

import numpy as np

cols = list(kpLogData.columns)
cols.remove('Team_Year')

for col in cols:
    col_exp = col + '_exp'
    kpLogData[col_exp] = np.exp(kpLogData[col])
    
## Pull out only useful exp tranformed variables
kpExp = kpLogData[['Team_Year', 'Pyth_NCSOS_exp', 'Luck_exp']]
kpExp.info()

## Take log of all variables

cols = list(kpLogData.columns)
cols.remove('Team_Year')

for col in cols:
    col_log = col + '_log'
    kpLogData[col_log] = np.log(kpLogData[col])
    
## Pull out only useful log transforms 
kpLog = kpLogData[['Team_Year','AdjO_Vol_Range_log', 'AdjD_Vol_Range_log', 'AdjS_Vol_Range_log',
                  'AdjO_Vol_Avg_log', 'AdjD_Vol_Avg_log', 'AdjS_Vol_Avg_log']]
kpLog.info()

kpTransformed = kpMerge.merge(kpExp, on = 'Team_Year', how = 'left')
kpMergeFinal = kpTransformed.merge(kpLog, on = 'Team_Year', how = 'left')
kpMergeFinal.drop(['Luck_zscore', 'Pyth_NCSOS_zscore' ,'AdjO_Vol_Avg_zscore' ,'AdjD_Vol_Avg_zscore',
                  'AdjS_Vol_Avg_zscore', 'AdjO_Vol_Range_zscore', 'AdjD_Vol_Range_zscore', 'AdjS_Vol_Range_zscore']
                 ,inplace=True,axis=1,errors='ignore')
kpMergeFinal.info()

regUse = regSeasonComp[['Wteam', 'Lteam', 'Home', 'Neutral', 'Away', 'Win']]
tourneyUse = tourneyResults[['Wteam', 'Lteam', 'Home', 'Neutral', 'Away', 'Win']]

allUse = pd.concat([regUse, tourneyUse])

kpModel = allUse.merge(kpMergeFinal, left_on = 'Wteam', right_on = 'Team_Year')
kpModel = kpModel.merge(kpMergeFinal, left_on = 'Lteam', right_on = 'Team_Year')

kpModel['Pyth_diff'] = kpModel['Pyth_zscore_x'] - kpModel['Pyth_zscore_y']
kpModel['AdjO_diff'] = kpModel['AdjO_zscore_x'] - kpModel['AdjO_zscore_y']
kpModel['AdjD_diff'] = kpModel['AdjD_zscore_x'] - kpModel['AdjD_zscore_y']
kpModel['AdjS_diff'] = kpModel['AdjS_zscore_x'] - kpModel['AdjS_zscore_y']
kpModel['AdjT_diff'] = kpModel['AdjT_zscore_x'] - kpModel['AdjT_zscore_y']
kpModel['Luck_diff'] = kpModel['Luck_exp_x'] - kpModel['Luck_exp_y']
kpModel['Pyth_SOS_diff'] = kpModel['Pyth_SOS_zscore_x'] - kpModel['Pyth_SOS_zscore_y']
kpModel['OppO_diff'] = kpModel['OppO_zscore_x'] - kpModel['OppO_zscore_y']
kpModel['OppD_diff'] = kpModel['OppD_zscore_x'] - kpModel['OppD_zscore_y']
kpModel['Pyth_NCSOS_diff'] = kpModel['Pyth_NCSOS_exp_x'] - kpModel['Pyth_NCSOS_exp_y']
kpModel['AdjO_Vol_Max_diff'] = kpModel['AdjO_Vol_Max_zscore_x'] - kpModel['AdjO_Vol_Max_zscore_y']
kpModel['AdjD_Vol_Max_diff'] = kpModel['AdjD_Vol_Max_zscore_x'] - kpModel['AdjD_Vol_Max_zscore_y']
kpModel['AdjS_Vol_Max_diff'] = kpModel['AdjS_Vol_Max_zscore_x'] - kpModel['AdjS_Vol_Max_zscore_y']
kpModel['AdjO_Vol_Min_diff'] = kpModel['AdjO_Vol_Min_zscore_x'] - kpModel['AdjO_Vol_Min_zscore_y']
kpModel['AdjD_Vol_Min_diff'] = kpModel['AdjD_Vol_Min_zscore_x'] - kpModel['AdjD_Vol_Min_zscore_y']
kpModel['AdjS_Vol_Min_diff'] = kpModel['AdjS_Vol_Min_zscore_x'] - kpModel['AdjS_Vol_Min_zscore_y']
kpModel['AdjO_Vol_Range_diff'] = kpModel['AdjO_Vol_Range_log_x'] - kpModel['AdjO_Vol_Range_log_y']
kpModel['AdjD_Vol_Range_diff'] = kpModel['AdjD_Vol_Range_log_x'] - kpModel['AdjD_Vol_Range_log_y']
kpModel['AdjS_Vol_Range_diff'] = kpModel['AdjS_Vol_Range_log_x'] - kpModel['AdjS_Vol_Range_log_y']
kpModel['AdjO_Vol_Avg_diff'] = kpModel['AdjO_Vol_Avg_log_x'] - kpModel['AdjO_Vol_Avg_log_y']
kpModel['AdjD_Vol_Avg_diff'] = kpModel['AdjD_Vol_Avg_log_x'] - kpModel['AdjD_Vol_Avg_log_y']
kpModel['AdjS_Vol_Avg_diff'] = kpModel['AdjS_Vol_Avg_log_x'] - kpModel['AdjS_Vol_Avg_log_y']

kpModel['id'] = kpModel.Wteam.str[:4] + '_' + kpModel.Wteam.str[4:] + '_' + kpModel.Lteam.str[4:]
kpFinal = kpModel.iloc[:,[2,3,4,5,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74]]
Target = 'Win'
IDcol = 'id'
kpFinal.info()

import sklearn.ensemble as ens
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from matplotlib.pylab import rcParams
import numpy as np
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 12,4

tIndex = np.random.rand(len(kpFinal)) < 0.75
X_train = kpFinal[tIndex]
X_test = kpFinal[~tIndex]

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds =5):
    ## Fit algorithm to the data
    alg.fit(dtrain[predictors], dtrain['Win'])
    ## Predict Training Set
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob=alg.predict_proba(dtrain[predictors])[:,1]
    ## Perform Cross validation
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Win'], 
                                                    cv=cv_folds, scoring = 'roc_auc')
    ## Print model report
    print("\nModel Report")
    print("Accuracy: %.4g" % metrics.accuracy_score(dtrain['Win'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Win'], dtrain_predprob))
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" 
        % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
        
    ## Print Feature Importance
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature_Importances')
        plt.ylabel('Feature_Importance_Score')
        
predictors = [x for x in X_train.columns if x not in [Target, IDcol]]

gbmFin = ens.GradientBoostingClassifier(max_depth = 7 , learning_rate = 0.025, min_samples_split = 250,
                                          min_samples_leaf = 50, max_features = 'sqrt', subsample= .80,
                                         random_state = 10, n_estimators = 1000)
modelfit(gbmFin, X_train, predictors, performCV = False)
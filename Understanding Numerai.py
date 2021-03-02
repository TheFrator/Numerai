# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:47:17 2021

@author: nicho
"""

import matplotlib
import numpy
import pandas as pd
import random
import sklearn
import xgboost
import matplotlib.pyplot as plt
# import dependencies
import pandas
import numerapi
import sklearn.linear_model


from sklearn import (
    feature_extraction, feature_selection, decomposition, linear_model,
    model_selection, metrics, svm
)

#pandas.options.display.max_rows=1000
#pandas.options.display.max_columns=300
#%%  Video 1 of Arbitrage's series


df_train = pd.read_csv("numerai_training_data.csv",nrows=50000)
df_train.info() #huh pretty powerful function for describing a dataframe without a variable explorer
y = df_train.target.values
df_train.target.unique()
df_train.target.describe() #difficult puzzle because there's no outliers mean == median. As humans there isnt muc we can interpret


features = [c for c in df_train if c.startswith("feature")]

df_train[features].head(10) #prints first 10 rows of the dataframe - includes every column


#looking at 1 era

era1 = df_train[df_train.era == 'era1'].copy() #eras are time components but theyre undefined months?days?
era1.describe() #Nothing can be inferred!!!




#%%

df=pd.read_csv("numerai_training_data.csv")


tournament_data =  pd.read_csv("numerai_tournament_data.csv")
df.head()

# There's 310 features
features = [c for c in df if c.startswith("feature")]
df["erano"] = df.era.str.slice(3).astype(int)
eras = df.erano
target = "target"
len(features)


feature_groups = {
    g: [c for c in df if c.startswith(f"feature_{g}")]
    for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
}

#feature_groups


# The models should be scored based on the rank-correlation (spearman) with the target
def numerai_score(y_true, y_pred):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    return numpy.corrcoef(y_true, rank_pred)[0,1]

# It can also be convenient while working to evaluate based on the regular (pearson) correlation
def correlation_score(y_true, y_pred):
    return numpy.corrcoef(y_true, y_pred)[0,1]


# There are 120 eras numbered from 1 to 120
eras.describe()

# The earlier eras are smaller, but generally each era is 4000-5000 rows
df.groupby(eras).size().plot()
#df.groupby(eras).describe() 

# The target is discrete and takes on 5 different values
df.groupby(target).size()

#%%
# Some of the features are very correlated
#Especially within feature groups

feature_corrs = df[features].corr()

feature_corrs.stack().head()

tdf = feature_corrs.stack()
tdf = tdf[tdf.index.get_level_values(0) < tdf.index.get_level_values(1)]
tdf.sort_values()

### The correlation can change over time
#You can see this by comparing feature correlations on the first half and second half on the training set

df1 = df[eras<=eras.median()]
df2 = df[eras>eras.median()]

corr1 = df1[features].corr().unstack()
corr1 = corr1[corr1.index.get_level_values(0) < corr1.index.get_level_values(1)]


corr2 = df2[features].corr().unstack()
corr2 = corr2[corr2.index.get_level_values(0) < corr2.index.get_level_values(1)]

tdf = pandas.DataFrame({
    "corr1": corr1,
    "corr2": corr2,
})
tdf["corr_diff"] = tdf.corr2 - tdf.corr1
tdf.sort_values(by="corr_diff")


#%%
## Some features are predictive on their own

feature_scores = {
    feature: numerai_score(df[target], df[feature])
    for feature in features
}

pandas.Series(feature_scores).sort_values()

# Single features do not work consistently though
by_era_correlation = pandas.Series({
    era: numpy.corrcoef(tdf[target], tdf["feature_strength34"])[0,1]
    for era, tdf in df.groupby(eras)
})
by_era_correlation.plot()

# With a rolling 10 era average you can see some trends
by_era_correlation.rolling(10).mean().plot()
#%% 

# Example way to upload my predicitons
# find only the feature columns
feature_cols = df.columns[df.columns.str.startswith('feature')]
# select those columns out of the training dataset
training_features = df[feature_cols]

# create a model and fit the training data (~30 sec to run)
model = sklearn.linear_model.LinearRegression()
model.fit(training_features, df.target)

# select the feature columns from the tournament data
live_features = tournament_data[feature_cols]
# predict the target on the live features
predictions = model.predict(live_features)

# predictions must have an `id` column and a `prediction_kazutsugi` column
predictions_df = tournament_data["id"].to_frame()
predictions_df["prediction_kazutsugi"] = predictions
predictions_df.head()

# Get your API keys and model_id from https://numer.ai/submit
public_id = "AAP7UFK3R2W2PS2GVU7MO5WTMPII2SER"
secret_key = "3AXY3QDUP2EKJJ4ZCDPJYSSVED5EJKBGEQJZ7KF4NMCINNXEZNQULZAPDM435G4N"
model_id = "265ea585-1af4-4e36-85fc-e6599054b7e8"
napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)

# Upload your predictions
predictions_df.to_csv("predictions.csv", index=False)
submission_id = napi.upload_predictions("predictions.csv", model_id=model_id)
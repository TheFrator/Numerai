
# %%
import pandas as pd
import numpy as np
import hts

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")
# %%
df=pd.read_csv("numerai_training_data.csv")
print(df.head())

#map the feature groups to a dicionary
feature_groups = {
    g: [c for c in df if c.startswith(f"feature_{g}")]

    for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
}
feature_groups
print(feature_groups.keys())
# %%
df_bottom_level = df.copy()


df_middle_level = df.copy()
df_middle_level.columns = df_middle_level.columns.str.replace(r"[0-9]+", "")
df_middle_level.columns = df_middle_level.columns.str.replace("feature_", "")
df_middle_level.columns
# %%
#df_middle_level = df_middle_level.mean(axis=1, level=0, skipna = True)
df_middle_level

df_middle_level = df_middle_level.groupby(level=0, axis=1).sum()


df_middle_level = df_middle_level.groupby(["era", "intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]) \
                    .sum() \
                    .reset_index(drop=False) \
                    .pivot(index="era", columns=["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]\
                        , values="target")
#df['grouped_feature'] = df_middle_level.apply(lambda x:np.mean([x[c] for c in feature_groups.keys() if c.split('_'[1])]))
# %%
df_middle_level.to_csv('df_middle_level.csv')
# %%

#Eventually did more research and determined that with the Numerai dataset, HTS models 
# would not be great 
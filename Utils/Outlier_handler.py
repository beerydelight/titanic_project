from scipy.stats import skew
import pandas as pd
import numpy as np
from Utils.utils import cleaned_one_hot as df
import seaborn as sns
import matplotlib.pyplot as plt
def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                         np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df



for col in ['age', 'fare', 'sibsp', 'parch']:
    print(f"{col}: {skew(df[col]):.2f}")

for col in ['age', 'fare', 'sibsp', 'parch']:
    if col in df.columns:
        df = cap_outliers_iqr(df, col)

# 2. Transform skewed features
df['fare_log'] = np.log1p(df['fare'])
df['sibsp_sqrt'] = np.sqrt(df['sibsp'])
df['parch_sqrt'] = np.sqrt(df['parch'])

# Drop originals
df = df.drop(columns=['fare', 'sibsp', 'parch'])

df.to_csv("/home/amine_pc/PyCharmMiscProject/Python M1/Python_Avance/Project_titanic/data/final.csv")
outlier_handled = df

'''
for col in ['age', 'fare_log', 'sibsp_sqrt', 'parch_sqrt']:
    print(f"{col}: {skew(df[col]):.2f}")

sns.histplot(df['fare_log'], kde=True)
plt.title('Fare after log transformation')
plt.show()
'''
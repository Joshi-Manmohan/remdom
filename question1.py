#Write a python program to carry out preprocessing of datasets such as encountering & handling missing values.

import numpy as np
import pandas as pd
num_rows = 100
num_classes = 4
col1 = np.random.randint(0, num_classes, num_rows)
col2 = np.random.randint(0, num_classes, num_rows)
col3 = np.random.randint(0, num_classes, num_rows)
col4 = np.random.randint(0, num_classes, num_rows)
col5 = np.random.randint(0, num_classes, num_rows)
data = pd.DataFrame({
'Column1': col1,
'Column2': col2,
'Column3': col3,
'Column4': col4, 'Column5':col5})
data_dropped=data.dropna()
print(data_dropped)
df_mean_filled = data.fillna(data.mean())
df_median_filled = data.fillna(data.median())
df_ffilled = data.ffill()
df_bfilled = data.bfill()
print("Original DataFrame:")
print(data)
print("\nDataFrame after removing rows with missing values:")
print(data_dropped)
print("\nDataFrame after filling missing values with column means:")
print(df_mean_filled)
print("\nDataFrame after filling missing values with column medians:")
print(df_median_filled)
print("\nDataFrame after forward filling missing values:")
print(df_ffilled)
print("\nDataFrame after backward filling missing values:")
print(df_bfilled)

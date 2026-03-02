import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense



print('\n')
data_path = 'data/csv/MachineLearningCVE'
df_list = []

for file in os.listdir(data_path):
    file_path = os.path.join(data_path, file)
    df_file = pd.read_csv(file_path)
    df_list.append(df_file)
    print(file_path)

df_complete = pd.concat(df_list, ignore_index=True)
print(df_complete)
# print(df_complete.columns)

df_benign = df_complete[df_complete[' Label'] == 'BENIGN']
df_malign = df_complete[df_complete[' Label'] != 'BENIGN']
print('Porcentaje Benignos: ', len(df_benign) / len(df_complete))
print('Porcentaje Malignos: ', len(df_malign) / len(df_complete))
# print(df_malign[' Label'].value_counts())
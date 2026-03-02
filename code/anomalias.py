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

# Hacemos una pequeña limpieza. Eliminaoms la columna label porque la red solo entiende valores numericos, y eliminamos algunos valores que aparecen como infinitos en el df
X_benign = df_benign.drop(columns=[' Label'])
X_benign = X_benign.replace([np.inf, -np.inf], np.nan).dropna()

# Dividimos los datos en 80% para entrenar el modelo y 20% para test
X_train, X_test = train_test_split(X_benign, test_size=0.20, random_state=42)
print('\n')
print('Tamaño de datos de entrenamiento: ', X_train.shape)
print('Tamaño de datos de test: ', X_test.shape)


# Normalizamos los datos para que tengan valores entre 0 y 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Creamos la red neuronal (autoencoder) para predecir los datos
input_dim = X_train_scaled.shape[1]
entrada = Input(shape=(input_dim,))
encoder = Dense(int(input_dim/2), activation='relu')(entrada)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=entrada, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Ahora es cuando entrenamos la red con los datos que hemos dividido anteriormente
print('Comenzamos el entrenamiento de la red...')
history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled, # Al ser un autoencoder la salida esperada tiene que ser igual a los inputs del modelo
    epochs=5,
    batch_size=256,
    validation_data=(X_test_scaled, X_test_scaled),
    verbose=1
)
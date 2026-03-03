import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense



print('\n')
data_path = 'data/csv/'
df_list = []

for file in os.listdir(data_path):
    # Comprobamos que el archivo sea csv, ya que tengo ademas el archivo .zip en esa carpeta
    if file.endswith('.csv'):
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



# Hacemos una pequeña limpieza
# Eliminaoms la columna label porque la red solo entiende valores numericos, y eliminamos algunos valores que aparecen como infinitos en el df
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



# A continuacion vamos a ver si se cumple la teoria. 
# Al pasar los datos de los logs malignos por la red neuronal, deberiamos obtener un error cuadratico medio (mse) mucho mayor, vamos a ver si esto es cierto

# Para ello tenemos que hacer un preprocesamiento de igual manera que para X_benign
# Eliminamos la columna ' Label', eliminamos valores infinitos y normalizamos los datos
X_malign = df_malign.drop(columns=[' Label'])
X_malign = X_malign.replace([np.inf, -np.inf], np.nan).dropna()
X_malign_scaled = scaler.transform(X_malign)

# Predecimos los resultados con el modelo (autoencoder) definido anteriormente
results_benign = autoencoder.predict(X_test_scaled)
results_malign = autoencoder.predict(X_malign_scaled)

# Calculamos el error cuadratico medio para cada caso y los comparamos
# Segun vimos en la teoria, el error cuadratico medio mse tiene que ser mucho mayor para los logs malignos, ya que la red esta entrenada unicamente con datos benignos, por lo que el autoencoder la tasa de error en este caso se dispararia
mse_benign = np.mean(np.power(X_test_scaled - results_benign, 2))
mse_malign = np.mean(np.power(X_malign_scaled - results_malign, 2))
print('MSE de logs benignos: ', mse_benign)
print('MSE de logs malignos: ', mse_malign)

# Con estos resultados el error de los datos malignos es aproximadamente 100 veces mas grande que los benignos, lo que supone una diferencia significativa y que cumple perfectamente con lo que esperabamos
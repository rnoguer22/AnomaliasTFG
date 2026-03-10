import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense





class Anomalias:

    def __init__(self):
        load_dotenv()
        # Usamos load_dotenv() para obtener las variables del archivo .env, para evitar filtrar la ruta completa del repositorio
        self.repo_path = os.getenv('REPOSITORY_PATH')
        self.data_path = os.path.join(self.repo_path, 'data/csv/')
        self.df_list = []
        self.scaler = MinMaxScaler()

    # Metodo para obtener un dataframe a partir de los datos del CICIDS2017
    def get_data(self):
        for file in os.listdir(self.data_path):
            # Comprobamos que el archivo sea csv, ya que tengo ademas el archivo .zip en la carpeta
            if file.endswith('.csv'):
                file_path = os.path.join(self.data_path, file)
                df_file = pd.read_csv(file_path)
                self.df_list.append(df_file)

        self.df = pd.concat(self.df_list, ignore_index=True)
        return self.df
    
    # Metodos para obtener los dataframes unicamente con datos benignos o malignos, y el scaler
    def get_df_benign(self):
        self.df_benign = self.df[self.df[' Label'] == 'BENIGN']
        return self.df_benign
    
    def get_df_malign(self):
        self.df_malign = self.df[self.df[' Label'] != 'BENIGN']
        return self.df_malign
    
    def get_scaler(self):
        return self.scaler
    


    # Metodo para obtener un gráfico de barras con la proporcion entre logs malignos y benignos
    def plot_class_distribution(self, save=True, malign=False):
        plt.figure(figsize=(12, 6))

        df_plot = self.df.copy()
        # Podemos mostrar unicamente los datos benignos vs malignos (sin categorizar los datos malignos), o mostrar solo los malignos (clasificados por tipo)
        if malign:
            df_plot = df_plot[df_plot[' Label'] != 'BENIGN']
            title = 'Distribución de Clases: Logs Malignos'
            filename = 'class_distribution_malign.png'
            plt.xticks(rotation=45)
        else:
            df_plot[' Label'] = (df_plot[' Label'] == 'BENIGN').map({True: 'BENIGN', False: 'MALIGN'})
            title = 'Distribución de Clases: Benignos vs Malignos'
            filename = 'class_distribution_benign_vs_malign.png'

        # Hacemos un countplot para ver el número de valores de cada opcion posible en la columna ' Label'
        sns.countplot(data=df_plot, x=' Label', order=df_plot[' Label'].value_counts().index, palette='Set2')
        plt.title(title)
        plt.ylabel('Número de logs')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.repo_path, 'img/code/', filename))
        plt.show()



    # Metodo para ver la matriz de correlacion
    def plot_correlation_matrix(self, n_features=20, save=True):
        # Seleccionamos solo las 'n' columnas con mayor varianza ya que son demasiadas columnas y no podemos ponerlas todas (para que el grafico sea legible, porque si ponemos las 80 columnas no veriamos nada)
        df_numeric = self.df.drop(columns=[' Label'])
        # Eliminamos columnas con varianza cero (constantes)
        df_numeric = df_numeric.loc[:, df_numeric.var() > 0]
        
        top_features = df_numeric.var().sort_values(ascending=False).head(n_features).index
        # Obtneemos la matriz con las distintas correlaciones
        corr_matrix = df_numeric[top_features].corr()

        # Mostramos la matriz en un grafico
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f'Matriz de Correlación (Top {n_features} variables con mayor varianza)')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.repo_path, 'img/code/matriz_correlacion.png'))
        plt.show()


    
    # Metodo para hacer una pequeña limpieza del dataset (ya que los datos estan previamente limpiados por el CIC)
    def clean_df(self, df_to_clean):
        # Eliminaoms la columna label porque la red solo entiende valores numericos, y eliminamos algunos valores que aparecen como infinitos en el df
        df_to_clean = df_to_clean.drop(columns=[' Label'])
        df_to_clean = df_to_clean.replace([np.inf, -np.inf], np.nan).dropna()
        return df_to_clean

    # Mtodo para dividir los datos en 80% para entrenar el modelo y 20% para test
    def get_scaled_train_test_data(self, df_to_scale):
        X_train, X_test = train_test_split(df_to_scale, test_size=0.20, random_state=42)
        # Normalizamos los datos para que tengan valores entre 0 y 1
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled



    # Metodo que crear, entrena y devuelve el autoencoder con el fin de predecir los datos
    def create_and_train_autoencoder(self, X_train_scaled, X_test_scaled):
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
        return autoencoder, history
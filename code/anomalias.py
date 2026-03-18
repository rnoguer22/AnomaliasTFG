import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
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
        self.model_path = os.path.join(self.repo_path, 'data/model')
        self.df_list = []
        self.drop_columns = []
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
    def get_df_benign(self, df):
        self.df_benign = df[df[' Label'] == 'BENIGN']
        return self.df_benign
    
    def get_df_malign(self, df):
        self.df_malign = df[df[' Label'] != 'BENIGN']
        return self.df_malign
    
    def get_scaler(self):
        return self.scaler
    
    

    # Metodo para obtener un dataframe con los valores atípicos de los datos
    def get_outliers_df(self, df):
        outliers_data = []
        for column in df.select_dtypes(include='number').columns:
            # Definimos los cuartiles y el rango intercuartilico
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            # Con los cuartiles y el rango intercuartilico podemos hallar los outliers
            lower_outliers = q1 - 1.5 * iqr
            upper_outliers = q3 + 1.5 * iqr
            # Filtramos en el dataset para encontrar los valores
            outliers_number = df[(df[column] < lower_outliers) | (df[column] > upper_outliers)].shape[0]
            # Añadimos los datos unicamente si detectamos valores atipicos
            if outliers_number > 0:
                outliers_perc = (outliers_number / len(df)) * 100
                outliers_data.append([column, outliers_number, outliers_perc])

        # Creamos el dataframe, ordenamos los datos de manera descendente y lo devolvemos
        df_outliers = pd.DataFrame(outliers_data, columns=['Columna', 'Número de Outliers', 'Porcentaje'])
        df_outliers.sort_values('Número de Outliers', ascending=False)
        return df_outliers    
    


    # Funcion para analizar los datos de una columna. Sacamos histograma, diagrama de caja y QQ plot
    def plot_normality_test(self, df, column_name):
        plt.figure(figsize=(15, 5))
        data = df[column_name]

        # Histograma
        plt.subplot(1, 3, 1)
        sns.histplot(data, kde=True, color='skyblue')
        plt.title(f'Histograma de {column_name}')

        # Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(y=data, color='lightgreen')
        plt.title(f'Boxplot de {column_name}')

        # Q-Q Plot
        plt.subplot(1, 3, 3)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot de {column_name}')

        plt.tight_layout()
        plt.savefig(os.path.join(self.repo_path, 'img/code', 'normality_test.png'))
        plt.show()
    


    # Metodo para obtener un gráfico de barras con la proporcion entre logs malignos y benignos
    def plot_class_distribution(self, save=True, malign=False):
        plt.figure(figsize=(12, 6))

        df_plot = self.df.copy()
        # Podemos mostrar unicamente los datos benignos vs malignos (sin categorizar los datos malignos), o mostrar solo los malignos (clasificados por tipo)
        if malign:
            df_plot = df_plot[df_plot[' Label'] != 'BENIGN']
            title = 'Distribución de Clases: Logs Malignos'
            filename = 'class_distribution_malign.png'
            plt.xticks(rotation=90)
        else:
            df_plot[' Label'] = (df_plot[' Label'] == 'BENIGN').map({True: 'BENIGN', False: 'MALIGN'})
            title = 'Distribución de Clases: Benignos vs Malignos'
            filename = 'class_distribution_benign_vs_malign.png'

        # Hacemos un countplot para ver el número de valores de cada opcion posible en la columna ' Label'
        sns.countplot(data=df_plot, x=' Label', order=df_plot[' Label'].value_counts().index, palette='bright')
        plt.title(title)
        plt.ylabel('Número de logs')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.repo_path, 'img/code/', filename))
        plt.show()


    def get_corr_matrix_from_df(self, df):
        # Seleccionamos solo las 'n' columnas con mayor varianza ya que son demasiadas columnas y no podemos ponerlas todas (para que el grafico sea legible, porque si ponemos las 80 columnas no veriamos nada)
        df_numeric = df.drop(columns=[' Label'])
        # Eliminamos columnas con varianza cero (constantes)
        df_numeric = df_numeric.loc[:, df_numeric.var() > 0]        
        # Obtneemos la matriz de correlacion
        corr_matrix = df_numeric.corr()
        return corr_matrix    



    # Metodo para ver la matriz de correlacion
    def plot_correlation_matrix(self, corr_matrix, filename, size, save=True):
        plt.figure(figsize=(size, size))
        sns.heatmap(corr_matrix, annot=False, cmap='viridis', square=True)
        plt.title(f'Matriz de Correlación')
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.repo_path, 'img/code', filename))
        plt.show()



    # Usamos otro metodo para limpiar las correlaciones que sean muy altas, y mostrar la matriz limpia
    def get_df_corr_cleaned(self, df, corr_matrix, umbral):
        # La matriz de correlacion es simetrica, asi que vamos a seleecionar uniamente el triangulo superior (podriamos elegir tambien el inferior)
        # k=0 es la diagonal, si queremos todo lo que hay por encima escogemos k=1. Si no, k=-1
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        drop_columns = []
        for column in upper_triangle.columns:
            # Al seleccionar el triangulo inferior nos aseguramos que solo se elimine una de los variables altamente correlacionadas entre si, y no ambas columnas
            if any(upper_triangle[column] >= umbral):
                drop_columns.append(column)

        # Eliminamos las columnas correspondientes
        print(f'Se eliminaran {len(drop_columns)} columnas: {drop_columns}')
        df = df.drop(drop_columns, axis=1)  
        return df
               




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



    # Metodo que crea, entrena y devuelve el autoencoder con el fin de replicar los datos
    def create_and_train_autoencoder(self, X_train_scaled, X_test_scaled, save=True):
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
            epochs=30,
            batch_size=256,
            validation_data=(X_test_scaled, X_test_scaled),
            verbose=1
        )

        if save:
            # Guardamos el modelo y el scaler para poder acceder a ellos mas facilmente
            joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))
            joblib.dump(autoencoder, os.path.join(self.model_path, 'autoencoder.pkl'))

        return autoencoder, history
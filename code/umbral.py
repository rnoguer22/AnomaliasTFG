import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv





class Umbral:

    def __init__(self):
        load_dotenv()
        self.repo_path = os.getenv('REPOSITORY_PATH')
        self.csv_path = os.path.join(self.repo_path, 'data/umbral/umbral_mse_fast.csv')       
        self.df = pd.read_csv(self.csv_path)
        self.mse_values = self.df['mse'].values



    # Metodo para calcular el umbral en funcion del metodo de la desviacion estandar
    # Dicho metodo calcula el umbral basado en media + k * desviacion estandar de los datos
    def metodo_desviacion_estandar(self, k=3):
        media = np.mean(self.mse_values)
        sigma = np.std(self.mse_values)
        umbral = media + (k * sigma)
        return round(umbral, 4), round(media, 4), round(sigma, 4)

    # Metodo para calcular el umbral en funcion del metodo de los percentiles
    # Con esto estamos ignorando el 0.01% (100 - p) de los datos que se encuentren en los extremos
    def metodo_percentil(self, p=99.9):
        umbral = np.percentile(self.mse_values, p)
        return round(umbral, 4), round(p, 4)

    # Ultimo metodo para definir el umbral, esta vez lo hacemos en funcion del valor maximo
    # Obtenemos el valor maximo del trafico de prueba tomado y le añadimos un margen de guarda
    def metodo_maximo_seguro(self, margen=1.0):
        maximo = np.max(self.mse_values)
        umbral = maximo + margen
        return round(umbral, 4), round(maximo, 4)





if __name__ == '__main__':

    umbral = Umbral()
    d_umbral, d_media, d_sigma = umbral.metodo_desviacion_estandar()
    print(f'Metodo desviacion estandar: {d_umbral}, {d_media}, {d_sigma}')
    p_umbral, p_percentil = umbral.metodo_percentil()
    print(f'Metodo percentiles: {p_percentil}: {p_umbral}')
    m_umbral, m_maximo = umbral.metodo_maximo_seguro()
    print(f'Metodo maximo: {m_maximo}: {m_umbral}')
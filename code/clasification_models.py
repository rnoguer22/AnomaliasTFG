import pandas as pd
import numpy as np
import joblib
import os
from config.columns import COLUMNS

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler





class Clasification_Model:

    def __init__(self):
        load_dotenv()
        self.repo_path = os.getenv('REPOSITORY_PATH')
        self.model_path = os.path.join(self.repo_path, 'data/model/')
        self.captured_model_path = os.path.join(self.model_path, 'captured/')
        self.df_path = os.path.join(self.repo_path, 'data/df')
        self.scaler_class = StandardScaler()
        #self.df_malign = pd.read_csv(os.path.join(self.df_path, 'df_malign_cleaned.csv'))
        self.df_malign_captured = pd.read_csv(os.path.join(self.df_path, 'df_malign_captured.csv'))
        self.label_encoder = LabelEncoder()
        


    # Creamos una funcion auxiliar para evaluar los modelos y ver datos como la accuracy, precision, f1, etc.
    # Este metodo simplemente nos ayuda a ahorrar lineas de codigo
    def predict_save_model(self, model, X_test, y_test):
        # Para ello tenemos que predecir los datos de test con el modelo seleccionado y mostrar las metricas
        y_pred = model.predict(X_test)
        model_name = model.__class__.__name__

        print(f"\n{'='*40}")
        print(f" RESULTADOS: {model_name} ")
        print(f"{'='*40}")
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            
        # Finalmente guardamos el modelo en la carpeta correspondiente
        joblib.dump(model, os.path.join(self.captured_model_path, f'{model_name}.pkl'))
        print(f"[*] Modelo guardado en: {os.path.join(self.captured_model_path, f"{model_name}.pkl")}")

    

    # A continuacion tenemos las distintas funciones para entrenar los diferentes modelos que se van a evaluar
    def train_rf(self, X_train, y_train, X_test, y_test):
        print("\n[*] Entrenando Random Forest...")
        rf = RandomForestClassifier(
                n_estimators=300, 
                max_depth=20, 
                min_samples_split=5,
                random_state=42, 
                n_jobs=-1
            )
        rf.fit(X_train, y_train)
        self.predict_save_model(rf, X_test, y_test)


    def train_xgb(self, X_train, y_train, X_test, y_test):
        print("\n[*] Entrenando XGBoost...")
        num_class = len(np.unique(y_train))
        xgb = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42, 
            eval_metric='mlogloss',
            objective='multi:softprob', # añadimos este campo para indicar que es una clasificacion con multiples clases
            num_class=num_class)
        
        xgb.fit(X_train, y_train)
        self.predict_save_model(xgb, X_test, y_test)


    def train_knn(self, X_train, y_train, X_test, y_test):
        print("\n[*] Entrenando K-Nearest Neighbors...")
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_train, y_train)
        self.predict_save_model(knn, X_test, y_test)


    def train_mlp(self, X_train, y_train, X_test, y_test):
        print("\n[*] Entrenando Perceptrón Multicapa (Deep Learning)...")
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
        mlp.fit(X_train, y_train)
        self.predict_save_model(mlp, X_test, y_test)


    # Y por ultimo utilizamos un metodo para evaluar una entrada de datos (capturada en tiempo real en stream_detection.py)
    # con el fin de clasificar el tipo de ataque y evaluar las metricas, para optar por el modelo que mejor se ajuste a los datos de test
    def predict_stream_flow(self, stream_flow, model = ''):   
        #scaled_stream_flow = self.scaler.transform(stream_flow)

        # Obtenemos el label_encoder que se uso previamente para entrenar los modelos
        map_encoder_file = joblib.load(os.path.join(self.model_path, 'label_encoder.pkl'))
        # Ahora tenemos que generar el encoder inverso, en lugar de {0: 'DoS', 1: ...} necesitamos {'DoS': 0, ...}
        # ya que ahora vamos en el sentido contrario, necesitamos obtener el string en funcion del valor numerico
        revert_map_encoder_dict = {v: k for k, v in map_encoder_file.items()}
                
        models = ['RandomForestClassifier', 'XGBClassifier', 'KNeighborsClassifier', 'MLPClassifier']
        if model in models:
            model_name, prediction_text, confianza = self.get_model_results(model, stream_flow, revert_map_encoder_dict)  
            print(f" -> {model_name.ljust(15)} : {prediction_text} (Confianza: {confianza:.2f}%)")
            return model_name, prediction_text, confianza

        else: 
            result_models_list = []   
            for model in models:
                model_name, prediction_text, confianza = self.get_model_results(model, stream_flow, revert_map_encoder_dict)  
                result_models_list.append([model_name, prediction_text, confianza])
                print(f" -> {model_name.ljust(15)} : {prediction_text} (Confianza: {confianza:.2f}%)")
            return result_models_list    


    # Metodo para obtener los resultados de la prediccion del modelo
    def get_model_results(self, model_name, stream_flow, revert_map_encoder_dict):
        # Comprobamos que los modelos existen antes de cargarlos directamente en memoria
        model_file_path = os.path.join(self.captured_model_path, f'{model_name}.pkl')
        if os.path.exists(model_file_path):
            model = joblib.load(model_file_path)

            # Realizamos la prediccion con el modelo, esta clasificacion sera un numero, por lo que tenemos que aplicar el label_encoder inverso que obtuvimos antes
            prediction = model.predict(stream_flow)[0]
            prediction_text = revert_map_encoder_dict.get(prediction, f"ID: {prediction}")
            
            # Obtenemos la confianza del resultado de la prediccion
            probs = model.predict_proba(stream_flow)[0]
            confianza = max(probs) * 100

            return model_name, prediction_text, confianza
                            




if __name__ == "__main__":  

    clasif = Clasification_Model()  

    # Obtenemos los datos de train y test de los logs con intenciones maliciosas, ya que son este tipo de anomalias las que queremos clasificar
    X = clasif.df_malign_captured.drop(" Label", axis=1)
    y = clasif.df_malign_captured[" Label"]  

    # Tenemos que convertir X con las columnas finales para el entrenamiento del random forest
    # Esto es porque inicialmente el modelo no predice bien, y un factor determinante es la redundancia de variables en los datos
    X = X[COLUMNS]    

    # Los modelos boosting necesitan que todos los valores sean numericos (La columna ' Label' contiene el tipo de ataque, en formato string)
    # Por ello tenemos que hacer un mapping y asignarle un numero a cada valor de ' Label'. Para ellos usamos un LabelEncoder clasico
    y_encoded = clasif.label_encoder.fit_transform(y)
    # Guardamos el mapeo de clases, ya que lo necesitaremos usar cuando obtengamos la prediccion en stream_detection.py
    # De esta manera veremos el valor del string, y no el valor asociado por el encoder, para que sea mas legible
    map = dict(zip(clasif.label_encoder.classes_, clasif.label_encoder.transform(clasif.label_encoder.classes_)))
    joblib.dump(map, os.path.join(clasif.model_path, 'label_encoder.pkl'))
    print(f"[*] Mapeo de clases generado: {map}")

    # Ahora si, procedemos con el train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    # Escalamos los datos, esta vez con un scaler distinto al usado para el autoencoder, ya que ahora si nos interesa la columna ' Label' al tratarse de modelos de clasificacion
    X_train_scaled = clasif.scaler_class.fit_transform(X_train)
    X_test_scaled = clasif.scaler_class.transform(X_test)

    # Guardamos el scaler
    joblib.dump(clasif.scaler_class, os.path.join(clasif.model_path, 'scaler_class.pkl'))

    # Entrenamos los distintos modelos
    clasif.train_rf(X_train_scaled, y_train, X_test_scaled, y_test)
    '''
    clasif.train_xgb(X_train_scaled, y_train, X_test_scaled, y_test)
    clasif.train_knn(X_train_scaled, y_train, X_test_scaled, y_test)
    clasif.train_mlp(X_train_scaled, y_train, X_test_scaled, y_test)
    '''
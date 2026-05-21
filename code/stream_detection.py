from nfstream import NFStreamer
import pandas as pd
import numpy as np
import joblib
import os
import csv
import requests
import subprocess
import socket
from dotenv import load_dotenv
from time import sleep

from umbral import Umbral
from config.columns import COLUMNS
from clasification_models import Clasification_Model





class Detection:

    def __init__(self):
        load_dotenv()
        self.repo_path = os.getenv('REPOSITORY_PATH')
        self.model_path = os.path.join(self.repo_path, 'data/model')
        self.csv_mse_path = os.getenv('CSV_MSE_PATH')
        self.csv_class_captured_path = os.getenv('CSV_CLASS_CAPTURED_PATH')

        # Leemos tanto el scaler como el model que obtenemos a partir de la clase Anomalias (anomalias.py)
        self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.pkl'))
        self.scaler_class = None
        self.autoencoder = joblib.load(os.path.join(self.model_path, 'autoencoder.pkl'))
        print(self.autoencoder.summary())

        # Definimos las columnas que debe tener el flujo de datos
        # Estas se encuentran definidas en el fichero code/config/columns.py
        self.columns = COLUMNS    

        # Definimos las variables para acceder a la api de telegram
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('CHAT_ID')



    # Metodo para mapear el flujo de datos que nos da nfstream al formato que tienen los dataframes con los datos de CICIDS2017
    def map_flow(self, flow):
        tot_fwd_pkts = flow.src2dst_packets
        tot_bwd_pkts = flow.dst2src_packets
        tot_pkts = flow.bidirectional_packets

        # Mapeamos calculando máximos, mínimos y promedios del bloque        
        row = {            
            ' Total Fwd Packets': tot_fwd_pkts,
            'Total Length of Fwd Packets': flow.src2dst_bytes,

            ' Fwd Packet Length Max': flow.src2dst_max_ps,
            ' Fwd Packet Length Min': flow.src2dst_min_ps,
            ' Fwd Packet Length Mean': (flow.src2dst_bytes / tot_fwd_pkts) if tot_fwd_pkts > 0 else 0,
            'Bwd Packet Length Max': flow.dst2src_max_ps,
            ' Bwd Packet Length Min': flow.dst2src_min_ps,

            # Flags TCP (Se mapean directamente desde el flujo único)
            'Fwd PSH Flags': 1 if flow.src2dst_psh_packets > 0 else 0,
            ' Bwd PSH Flags': 1 if flow.dst2src_psh_packets > 0 else 0,
            ' Fwd URG Flags': 1 if flow.src2dst_urg_packets > 0 else 0,
            ' Bwd URG Flags': 1 if flow.dst2src_urg_packets > 0 else 0,
            'FIN Flag Count': 1 if flow.bidirectional_fin_packets > 0 else 0,
            ' SYN Flag Count': 1 if flow.bidirectional_syn_packets > 0 else 0,
            ' RST Flag Count': 1 if flow.bidirectional_rst_packets > 0 else 0,
            ' PSH Flag Count': 1 if flow.bidirectional_psh_packets > 0 else 0,
            ' ACK Flag Count': 1 if flow.bidirectional_ack_packets > 0 else 0,
            ' URG Flag Count': 1 if flow.bidirectional_urg_packets > 0 else 0,

            # Tamaños y Varianza global
            ' Min Packet Length': flow.bidirectional_min_ps,
            ' Max Packet Length': flow.bidirectional_max_ps,
            ' Packet Length Mean': (flow.bidirectional_bytes / tot_pkts) if tot_pkts > 0 else 0,
            ' Packet Length Variance': flow.bidirectional_stddev_ps**2, 
            
            # Otros campos
            ' Down/Up Ratio': (tot_bwd_pkts / tot_fwd_pkts) if tot_fwd_pkts > 0 else 0,
        }

        # Creamos el DataFrame con los datos ya mapeados
        df_live_data_flow = pd.DataFrame([row])
        
        # Si faltase alguna columna, lanzamos un error
        for col in self.columns:
            if col not in df_live_data_flow.columns:
                raise  KeyError(
                    f'KeyError: No se encuentra la columna "{col}"'
                )  
    
        # Por ultimo solo nos queda reordenar el dataframe para que el modelo no confunda el orden de las variables y devolvemos el dataframe        
        return df_live_data_flow[self.columns] 



    # Este metodo es el que lo hace todo, obtiene el flujo de datos, los pasa por el autoencoder, predice el tipo de ataque, y manda la alerta por telegram
    def get_and_predict_live_flow(self):
        # Creamos el objeto de nfstream que nos captura los flujos de datos
        nfstreamer = NFStreamer(
                            source="any",
                            statistical_analysis=True, # Muy importante para capturar metricas estadisticas de la red (si no, faltarian todavia mas variables para convertir los datos)
                            splt_analysis=0,
                            bpf_filter=f"port 80",
                            promiscuous_mode=True,
                            idle_timeout=15, 
                            active_timeout=120)

        print("Iniciando rastreo de datos en tiempo real...")

        try:
            # Entramos en un bucle hasta que se detecte algun flujo, o hasta que el usuario detenga el programa
            for flow in nfstreamer:
                if 0 <= flow.dst_port < 1024 or flow.dst_port == 5050:   
                    
                    # Evitamos flujos rotos o trafico residual
                    # if flow.bidirectional_packets < 4 and flow.bidirectional_bytes < 200:
                        # continue

                    # Llamamos al metodo map_flow para que los datos tengan el mismo formato que el dataset                
                    df_live = self.map_flow(flow)
                    print(f'\nFlujo completado {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} | Packets: {flow.bidirectional_packets}')

                    # Aplicamos el scaler para que los datos tengan la misma escala que los datos de entrenamiento
                    df_flow_scaled = self.scaler.transform(df_live)

                    # A continuacion pasamos el flujo de datos por el autoencoder
                    prediction = self.autoencoder.predict(df_flow_scaled)
                    # Y calculamos el error cuadratico medio MSE
                    mse = round(np.mean(np.power(df_flow_scaled - prediction, 2), axis=1)[0], 4)
                    print(mse)

                    # Guardamos los datos en un fichero .csv
                    # Hacemos esto para la deteccion del umbral 
                    # self.save_mse_data_csv(flow.src_ip, mse)

                    #self.save_class_data_csv(df_live, 'Bruteforce')
                    
                    umbral_instance = Umbral()
                    umbral = umbral_instance.metodo_desviacion_estandar()[0]
                    if mse >= umbral:
                        # self.save_class_data_csv(df_live, 'Bruteforce')

                        clasification = Clasification_Model(captured=True)
                        self.scaler_class = joblib.load(os.path.join(clasification.model_path, 'scaler_class.pkl'))
                        df_flow_scaled = self.scaler_class.transform(df_live)
                        
                        model_name, prediction_text, confianza = clasification.predict_stream_flow(df_flow_scaled, 'RandomForestClassifier')    
                        message = f'Ataque detectado! MSE: {mse}\n' \
                                  f'IP atacante --> {flow.src_ip}:{flow.src_port}\n' \
                                  f'{model_name.ljust(15)}: {prediction_text}\n'
                        print(message)
                        message += f'\n¿Desea bloquear {flow.src_ip}?'

                        # Y por ultimo gestionamos el envio de la alerta con telegram
                        self.send_telegram_alert(message)

                        # Esperamos a que el usuario responda al mensaje...
                        confirmacion = self.wait_for_user_response()

                        # Una vez responde, procedemos a bloquear la IP o no, dependiendo de su respuesta
                        if confirmacion:
                            print(f"[*] El usuario confirmó el ataque. Bloqueando {flow.src_ip}...")
                            self.block_ip(flow.src_ip)
                        else:
                            print("[*] El usuario descartó el ataque. Continuando detección...")
                            self.send_telegram_alert("Ataque descartado. El sistema continúa monitorizando...")


                else:
                    print(f'Obviando flujo en puerto residual: {flow.dst_port}')    

        except KeyboardInterrupt:
            print("Deteniendo rastreo...")



    # Metodo para guardar tanto la IP como el MSE en un fichero csv, lo cual es util para determinar el umbral
    def save_mse_data_csv(self, ip, mse):
        file_exists = os.path.isfile(self.csv_mse_path)
        with open(self.csv_mse_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Si el archivo no existe, escribimos el nombre de las columnas de los datos
                writer.writerow(['ip', 'mse'])
            
            # Escribimos los datos introducidos como argumentos del metodo en el fichero .csv
            writer.writerow([ip, mse])


    # Este otro metodo es para generar un fichero csv con datos malignos generados por nosotros mismos, para comprobar que el modelo predice correctamente, ya que los datos de entrenamiento son muy distintos de los estamos generando con Hydra, SQLMap, hping3, etc.
    def save_class_data_csv(self, df_row, classification_value, path=''):
        if path == '':
            path = self.csv_class_captured_path
            columns_csv = df_row.columns.to_list()
        else:
            columns_csv = pd.read_csv(self.csv_class_captured_path, nrows=0).columns.tolist()
            df_row = pd.DataFrame(df_row, columns=columns_csv[:-1])

        file_exists = os.path.isfile(path)
        with open(path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Si el archivo no existe, escribimos el nombre de las columnas de los datos
                columns_csv.append(' Label')
                writer.writerow(columns_csv)
            
            # Escribimos los datos introducidos como argumentos del metodo en el fichero .csv
            data_values = df_row.iloc[0].to_list()
            data_values.append(classification_value)
            writer.writerow(data_values)


    # Metodo para mostrar una alerta a traves de telegram
    def send_telegram_alert(self, message):
        # Nos conectamos a la api de telegram con el token del bot
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        # Definimos el mensaje
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_notification': False
        }
        
        try:
            # Y mandamos el mensaje, creando asi una alerta en nuestro dispositivo
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                print(f"Error enviando a Telegram: {response.text}")
        except Exception as e:
            print(f"Error de conexión con Telegram: {e}")


    # Metodo paar esperar la respuesta del usuario a traves de Telegram
    def wait_for_user_response(self):
        print("[*] Esperando validación del analista en Telegram...")
        last_update_id = -1
        
        # Limpiamos mensajes antiguos obteniendo el último ID
        updates_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
        try:
            init_res = requests.get(updates_url).json()
            if init_res["result"]:
                last_update_id = init_res["result"][-1]["update_id"]
        except: pass

        # Necesitamos un bucle infinito que espere a la respuesta del usuario
        while True:
            try:
                # Consultamos nuevos mensajes (offset garantiza que solo leemos lo nuevo)
                res = requests.get(f"{updates_url}?offset={last_update_id + 1}").json()
                for update in res.get("result", []):
                    last_update_id = update["update_id"]
                    if "message" in update and "text" in update["message"]:
                        # Leemos el mensaje
                        user_text = update["message"]["text"].lower().strip()
                        
                        # Devolvemos true o false dependiendo de la respuesta del usuario
                        if user_text in ['si', 'sí']:
                            return True
                        elif user_text == 'no':
                            return False
                        else:
                            self.send_telegram_alert("Introduzca una respuesta válida: [Si, No]...")

            except Exception as e:
                print(f"[!] Error consultando Telegram: {e}")
            sleep(2)


    # Metodo para bloquear la IP del atacante
    def block_ip(self, ip):
        mi_ip = socket.gethostbyname(socket.gethostname())
        priviledges_ip = ['127.0.0.1', 'localhost', mi_ip]

        # Comprobamos que la IP a bloquear no es la ip de este dispositivo, para ahorrar problemas 
        if ip in priviledges_ip:
            message = f"[!] No se puede bloquear la IP {ip}, IP privilegiada"          
        else:    
            try:
                # Usamos iptables para bloquear todo el tráfico de esa IP
                subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'], check=True)
                message = f"[!] IP {ip} bloqueada permanentemente"
            except Exception as e:
                message = f"[!] Error al bloquear IP: {e}"

        print(message)
        self.send_telegram_alert(message)





if __name__ == '__main__':

    detection = Detection()
    detection.get_and_predict_live_flow()
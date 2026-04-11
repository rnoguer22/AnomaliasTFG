from nfstream import NFStreamer
import pandas as pd
import numpy as np
import joblib
import os
import time
import csv
from dotenv import load_dotenv

from umbral import Umbral





class Detection:

    def __init__(self):
        load_dotenv()
        self.repo_path = os.getenv('REPOSITORY_PATH')
        self.model_path = os.path.join(self.repo_path, 'data/model')
        self.csv_mse_path = os.getenv('CSV_MSE_PATH')
        # Leemos tanto el scaler como el model que obtenemos a partir de la clase Anomalias (anomalias.py)
        self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.pkl'))
        self.autoencoder = joblib.load(os.path.join(self.model_path, 'autoencoder.pkl'))
        print(self.autoencoder.summary())

        self.max_time = 10
        self.buffer_flow = {}

        # Definimos las columnas que debe tener el flujo de datos
        self.columns = [
            ' Destination Port',
            ' Flow Duration',
            ' Total Fwd Packets',
            'Total Length of Fwd Packets',
            ' Fwd Packet Length Max',
            ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean',
            'Bwd Packet Length Max',
            ' Bwd Packet Length Min',
            'Flow Bytes/s',
            ' Flow Packets/s',
            ' Flow IAT Mean',
            ' Flow IAT Std',
            ' Flow IAT Max',
            ' Flow IAT Min',
            ' Fwd IAT Mean',
            ' Fwd IAT Std',
            ' Fwd IAT Min',
            'Bwd IAT Total',
            ' Bwd IAT Mean',
            ' Bwd IAT Std',
            ' Bwd IAT Max',
            ' Bwd IAT Min',
            'Fwd PSH Flags',
            ' Bwd PSH Flags',
            ' Fwd URG Flags',
            ' Bwd URG Flags',
            'Fwd Packets/s',
            ' Bwd Packets/s',
            ' Min Packet Length',
            ' Max Packet Length',
            ' Packet Length Mean',
            ' Packet Length Variance',
            'FIN Flag Count',
            ' RST Flag Count',
            ' PSH Flag Count',
            ' ACK Flag Count',
            ' URG Flag Count',
            ' Down/Up Ratio',
            ' act_data_pkt_fwd',
            'Active Mean',
            ' Active Std',
            ' Active Max',
            ' Active Min',
            ' Idle Std'
            ]        



    # Metodo para mapear el flujo de datos que nos da nfstream al formato que tienen los dataframes con los datos de CICIDS2017
    def map_flow(self, flows, ip, port):
        print(f'Conviertiendo datos {ip}:{port}...')

        # 1. Agregamos las métricas sumables (Volumen y Duración)
        dur_ms = sum(f.bidirectional_duration_ms for f in flows)
        dur_sec = dur_ms / 1000.0 if dur_ms > 0 else 0.0001 # Evitar division por cero

        tot_fwd_pkts = sum(f.src2dst_packets for f in flows)
        tot_bwd_pkts = sum(f.dst2src_packets for f in flows)
        tot_fwd_bytes = sum(f.src2dst_bytes for f in flows)
        tot_bwd_bytes = sum(f.dst2src_bytes for f in flows)
        tot_pkts = sum(f.bidirectional_packets for f in flows)
        tot_bytes = sum(f.bidirectional_bytes for f in flows)

        # 2. Mapeamos calculando máximos, mínimos y promedios del bloque
        row = {
            # Tomamos el puerto del primer flujo como referencia (suelen apuntar al mismo)
            ' Destination Port': flows[0].dst_port, 
            ' Flow Duration': dur_ms * 1000,
            
            ' Total Fwd Packets': tot_fwd_pkts,
            'Total Length of Fwd Packets': tot_fwd_bytes,
            ' Fwd Packet Length Max': max(f.src2dst_max_ps for f in flows),
            ' Fwd Packet Length Min': min(f.src2dst_min_ps for f in flows),
            ' Fwd Packet Length Mean': (tot_fwd_bytes / tot_fwd_pkts) if tot_fwd_pkts > 0 else 0,
            
            'Bwd Packet Length Max': max(f.dst2src_max_ps for f in flows),
            ' Bwd Packet Length Min': min(f.dst2src_min_ps for f in flows),
            
            # Tasas (Bytes y Paquetes por segundo sobre la ventana total)
            'Flow Bytes/s': (tot_bytes / dur_sec),
            ' Flow Packets/s': (tot_pkts / dur_sec),
            'Fwd Packets/s': (tot_fwd_pkts / dur_sec),
            ' Bwd Packets/s': (tot_bwd_pkts / dur_sec),

            # Tiempos (Calculamos el promedio de los promedios como aproximación estadística)
            ' Flow IAT Mean': np.mean([f.bidirectional_mean_piat_ms for f in flows]) * 1000,
            ' Flow IAT Std': np.mean([f.bidirectional_stddev_piat_ms for f in flows]) * 1000,
            ' Flow IAT Max': max(f.bidirectional_max_piat_ms for f in flows) * 1000,
            ' Flow IAT Min': min(f.bidirectional_min_piat_ms for f in flows) * 1000,
            
            ' Fwd IAT Mean': np.mean([f.src2dst_mean_piat_ms for f in flows]) * 1000,
            ' Fwd IAT Std': np.mean([f.src2dst_stddev_piat_ms for f in flows]) * 1000,
            ' Fwd IAT Min': min(f.src2dst_min_piat_ms for f in flows) * 1000,
            
            'Bwd IAT Total': sum(f.dst2src_duration_ms for f in flows) * 1000,
            ' Bwd IAT Mean': np.mean([f.dst2src_mean_piat_ms for f in flows]) * 1000,
            ' Bwd IAT Std': np.mean([f.dst2src_stddev_piat_ms for f in flows]) * 1000,
            ' Bwd IAT Max': max(f.dst2src_max_piat_ms for f in flows) * 1000,
            ' Bwd IAT Min': min(f.dst2src_min_piat_ms for f in flows) * 1000,

            # Flags TCP (1 si CUALQUIER flujo en la ventana tiene esta flag)
            'Fwd PSH Flags': 1 if any(f.src2dst_psh_packets > 0 for f in flows) else 0,
            ' Bwd PSH Flags': 1 if any(f.dst2src_psh_packets > 0 for f in flows) else 0,
            ' Fwd URG Flags': 1 if any(f.src2dst_urg_packets > 0 for f in flows) else 0,
            ' Bwd URG Flags': 1 if any(f.dst2src_urg_packets > 0 for f in flows) else 0,
            'FIN Flag Count': 1 if any(f.bidirectional_fin_packets > 0 for f in flows) else 0,
            ' SYN Flag Count': 1 if any(f.bidirectional_syn_packets > 0 for f in flows) else 0,
            ' RST Flag Count': 1 if any(f.bidirectional_rst_packets > 0 for f in flows) else 0,
            ' PSH Flag Count': 1 if any(f.bidirectional_psh_packets > 0 for f in flows) else 0,
            ' ACK Flag Count': 1 if any(f.bidirectional_ack_packets > 0 for f in flows) else 0,
            ' URG Flag Count': 1 if any(f.bidirectional_urg_packets > 0 for f in flows) else 0,

            # Tamaños y Varianza global
            ' Min Packet Length': min(f.bidirectional_min_ps for f in flows),
            ' Max Packet Length': max(f.bidirectional_max_ps for f in flows),
            ' Packet Length Mean': (tot_bytes / tot_pkts) if tot_pkts > 0 else 0,
            ' Packet Length Variance': np.mean([f.bidirectional_stddev_ps**2 for f in flows]),
            
            # Otros campos
            ' Down/Up Ratio': (tot_bwd_pkts / tot_fwd_pkts) if tot_fwd_pkts > 0 else 0,
            ' act_data_pkt_fwd': tot_fwd_pkts,
            
            # Dejamos las variables Active a la duración total agregada
            'Active Mean': dur_ms * 1000,
            ' Active Std': 0,
            ' Active Max': dur_ms * 1000,
            ' Active Min': dur_ms * 1000,
            ' Idle Std': 0,
        }

        # Creamos el DataFrame con los datos ya mapeados
        df_live_data_flow = pd.DataFrame([row])
        
        # Si faltase alguna columna, la creamos y le asignamos un valor de 0 a todos sus registros
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
                            #bpf_filter=f"host {server_ip} and port 80",
                            bpf_filter=f"port 80",
                            promiscuous_mode=True,
                            idle_timeout=15, 
                            active_timeout=5)

        print("Iniciando rastreo de datos en tiempo real...")

        try:
            # Entramos en un bucle hasta que se detecte algun flujo, o hasta que el usuario detenga el programa
            for flow in nfstreamer:
                if 0 <= flow.dst_port < 1024 or flow.dst_port == 5050:
                    init_time = time.time()

                    if flow.src_ip not in self.buffer_flow:
                        self.buffer_flow[flow.src_ip] = {'flows': [], 'init_time': init_time}

                    self.buffer_flow[flow.src_ip]['flows'].append(flow)      

                    if init_time - self.buffer_flow[flow.src_ip]['init_time'] >= self.max_time:
                        flows = self.buffer_flow[flow.src_ip]['flows']

                        # Llamamos al metodo map_flow para que los datos tengan el mismo formato que el dataset                
                        df_live = self.map_flow(flows, flow.src_ip, flow.dst_port)

                        # Aplicamos el scaler para que los datos tengan la misma escala que los datos de entrenamiento
                        df_flow_scaled = self.scaler.transform(df_live)

                        # A continuacion pasamos el flujo de datos por el autoencoder
                        prediction = self.autoencoder.predict(df_flow_scaled)
                        # Y calculamos el error cuadratico medio MSE
                        mse = round(np.mean(np.power(df_flow_scaled - prediction, 2), axis=1)[0], 4)
                        #print('Numero de paquetes: ', df_live[' Total Fwd Packets'])
                        print('MSE: ', mse)

                        # Guardamos los datos en un fichero .csv
                        # Hacemos esto para la deteccion del umbral 
                        # self.save_data_csv(flow.src_ip, mse)

                        umbral_instance = Umbral()
                        umbral = umbral_instance.metodo_desviacion_estandar()[0]
                        if mse >= umbral:
                            print('Ataque detectado!')
                            print('Arrancando el sistema de prediccion de ataques...')



                        # Limpiamos el flujo del buffer para recibir nuevos datos
                        del self.buffer_flow[flow.src_ip]
                    
                else:
                    print(f'Obviando flujo en puerto residual: {flow.dst_port}')    

        except KeyboardInterrupt:
            print("Deteniendo rastreo...")



    # Metodo para guardar tanto la IP como el MSE en un fichero csv, lo cual es util para determinar el umbral
    def save_data_csv(self, ip, mse):
        file_exists = os.path.isfile(self.csv_mse_path)
        with open(self.csv_mse_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Si el archivo no existe, escribimos el nombre de las columnas de los datos
                writer.writerow(['ip', 'mse'])
            
            # Escribimos los datos introducidos como argumentos del metodo en el fichero .csv
            writer.writerow([ip, mse])





if __name__ == '__main__':

    detection = Detection()
    detection.get_and_predict_live_flow()
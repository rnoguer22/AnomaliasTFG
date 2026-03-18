from nfstream import NFStreamer
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv





class Detection:

    def __init__(self):
        load_dotenv()
        self.repo_path = os.getenv('REPOSITORY_PATH')
        self.model_path = os.path.join(self.repo_path, 'data/model')
        # Leemos tanto el scaler como el model que obtenemos a partir de la clase Anomalias (anomalias.py)
        self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.pkl'))
        self.autoencoder = joblib.load(os.path.join(self.model_path, 'autoencoder.pkl'))
        print(self.autoencoder.summary())

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
            ' Fwd Header Length',
            ' Bwd Header Length',
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
            'Fwd Avg Bytes/Bulk',
            ' Fwd Avg Packets/Bulk',
            ' Fwd Avg Bulk Rate',
            ' Bwd Avg Bytes/Bulk',
            ' Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate',
            'Init_Win_bytes_forward',
            ' Init_Win_bytes_backward',
            ' act_data_pkt_fwd',
            ' min_seg_size_forward',
            'Active Mean',
            ' Active Std',
            ' Active Max',
            ' Active Min',
            ' Idle Std'
            ]



    # Metodo para mapear el flujo de datos que nos da nfstream al formato que tienen los dataframes con los datos de CICIDS2017
    def map_flow(self, flow):
        print('Conviertiendo datos...')

        # Calculamos la duración en segundos para las tasas (evitando un error de division por cero)
        dur_sec = flow.bidirectional_duration_ms / 1000.0
        
        # Mapeamos el flujo de datos que se pasa como argumento a la funcion
        # De esta manera conseguimos que tenga el mismo formato que los datos descargados de CICIDS2017
        row = {
            ' Destination Port': flow.dst_port,
            ' Flow Duration': flow.bidirectional_duration_ms * 1000,
            ' Total Fwd Packets': flow.src2dst_packets,
            'Total Length of Fwd Packets': flow.src2dst_bytes,
            ' Fwd Packet Length Max': flow.src2dst_max_ps,
            ' Fwd Packet Length Min': flow.src2dst_min_ps,
            ' Fwd Packet Length Mean': flow.src2dst_mean_ps,
            'Bwd Packet Length Max': flow.dst2src_max_ps,
            ' Bwd Packet Length Min': flow.dst2src_min_ps,
            
            # Tasas (Bytes y Paquetes por segundo)
            'Flow Bytes/s': (flow.bidirectional_bytes / dur_sec) if dur_sec > 0 else 0,
            ' Flow Packets/s': (flow.bidirectional_packets / dur_sec) if dur_sec > 0 else 0,
            'Fwd Packets/s': (flow.src2dst_packets / dur_sec) if dur_sec > 0 else 0,
            ' Bwd Packets/s': (flow.dst2src_packets / dur_sec) if dur_sec > 0 else 0,

            # Tiempos entre paquetes (IAT) - Pasamos de ms a microsegundos
            ' Flow IAT Mean': flow.bidirectional_mean_piat_ms * 1000,
            ' Flow IAT Std': flow.bidirectional_stddev_piat_ms * 1000,
            ' Flow IAT Max': flow.bidirectional_max_piat_ms * 1000,
            ' Flow IAT Min': flow.bidirectional_min_piat_ms * 1000,
            ' Fwd IAT Mean': flow.src2dst_mean_piat_ms * 1000,
            ' Fwd IAT Std': flow.src2dst_stddev_piat_ms * 1000,
            ' Fwd IAT Min': flow.src2dst_min_piat_ms * 1000,
            'Bwd IAT Total': flow.dst2src_duration_ms * 1000,
            ' Bwd IAT Mean': flow.dst2src_mean_piat_ms * 1000,
            ' Bwd IAT Std': flow.dst2src_stddev_piat_ms * 1000,
            ' Bwd IAT Max': flow.dst2src_max_piat_ms * 1000,
            ' Bwd IAT Min': flow.dst2src_min_piat_ms * 1000,

            # Flags TCP (NFStream los da como conteo de paquetes)
            'Fwd PSH Flags': flow.src2dst_psh_packets,
            ' Bwd PSH Flags': flow.dst2src_psh_packets,
            ' Fwd URG Flags': flow.src2dst_urg_packets,
            ' Bwd URG Flags': flow.dst2src_urg_packets,
            'FIN Flag Count': flow.bidirectional_fin_packets,
            ' SYN Flag Count': flow.bidirectional_syn_packets,
            ' RST Flag Count': flow.bidirectional_rst_packets,
            ' PSH Flag Count': flow.bidirectional_psh_packets,
            ' ACK Flag Count': flow.bidirectional_ack_packets,
            ' URG Flag Count': flow.bidirectional_urg_packets,

            # Longitudes y Varianzas
            ' Fwd Header Length': flow.src2dst_packets * 40,
            ' Bwd Header Length': flow.dst2src_packets * 40,
            ' Min Packet Length': flow.bidirectional_min_ps,
            ' Max Packet Length': flow.bidirectional_max_ps,
            ' Packet Length Mean': flow.bidirectional_mean_ps,
            ' Packet Length Variance': (flow.bidirectional_stddev_ps)**2,
            
            # Otros campos
            ' Down/Up Ratio': (flow.dst2src_packets / flow.src2dst_packets) if flow.src2dst_packets > 0 else 0,
            ' act_data_pkt_fwd': flow.src2dst_packets,
            'Active Mean': flow.bidirectional_duration_ms * 1000,
            ' Active Std': 0,
            ' Active Max': flow.bidirectional_duration_ms * 1000,
            ' Active Min': flow.bidirectional_duration_ms * 1000,
            ' Idle Std': 0,

            # Campos que NFStream no extrae directamente (los inicializamos a 0 simplemente)
            'Init_Win_bytes_forward': 0,
            ' Init_Win_bytes_backward': 0,
            ' min_seg_size_forward': 0,
            'Fwd Avg Bytes/Bulk': 0, ' Fwd Avg Packets/Bulk': 0, ' Fwd Avg Bulk Rate': 0,
            ' Bwd Avg Bytes/Bulk': 0, ' Bwd Avg Packets/Bulk': 0, 'Bwd Avg Bulk Rate': 0,
        }

        # Creamos el DataFrame con los datos ya mapeados
        df_live_data_flow = pd.DataFrame([row])
        
        # Si faltase alguna columna, la creamos y le asignamos un valor de 0 a todos sus registros
        for col in self.columns:
            if col not in df_live_data_flow.columns:
                print(col)
                df_live_data_flow[col] = 0

        # Por ultimo solo nos queda reordenar el dataframe para que el modelo no confunda el orden de las variables y devolvemos el dataframe        
        return df_live_data_flow[self.columns] 



    # Este metodo es el que lo hace todo, obtiene el flujo de datos, los pasa por el autoencoder, predice el tipo de ataque, y manda la alerta por telegram
    def get_and_predict_live_flow(self, net_if, server_ip):
        # Creamos el objeto de nfstream que nos captura los flujos de datos
        nfstreamer = NFStreamer(source=f"{net_if}", 
                            statistical_analysis=True, # Muy importante para capturar metricas estadisticas de la red (si no, faltarian variables para convertir los datos)
                            splt_analysis=0,
                            bpf_filter=f"dst host {server_ip} and tcp dst port 8080",
                            promiscuous_mode=True,
                            # snapshot_len=100,
                            idle_timeout=15, 
                            active_timeout=120)

        print("Iniciando rastreo de datos en tiempo real...")

        try:
            # Entramos en un bucle hasta que se detecte algun flujo, o hasta que el usuario detenga el programa
            for flow in nfstreamer:
                print(flow)
                # Llamamos al metodo map_flow para que los datos tengan el mismo formato que el dataset                
                df_live = self.map_flow(flow)
                
                # Aplicamos el scaler para que los datos tengan la misma escala que los datos de entrenamiento
                df_flow_scaled = self.scaler.transform(df_live)

                # A continuacion pasamos el flujo de datos por el autoencoder
                prediction = self.autoencoder.predict(df_flow_scaled)
                # Y calculamos el error cuadratico medio MSE
                mse = np.mean(np.power(df_flow_scaled - prediction, 2), axis=1)[0]
                print(mse)
                
                # Y en caso de que se detectase como malicioso, realizar una clasificacion para obtener el tipo de ataque

        except KeyboardInterrupt:
            print("Deteniendo rastreo...")





if __name__ == '__main__':

    detection = Detection()
    detection.get_and_predict_live_flow('wlp2s0', '192.168.1.116')
# Del Bit al Comportamiento: Redefiniendo la frontera de la seguridad mediante el aprendizaje profundo

---

Este repositorio contiene la implementación práctica del Trabajo de Fin de Grado (TFG) enfocado en la **detección de anomalías y clasificación de ciberataques** en servidores web mediante técnicas de *Deep Learning* y *Machine Learning*.

---

## 📋 Índice
1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Estructura del Repositorio](#-estructura-del-repositorio)
3. [Despliegue e Implementación](#-despliegue-e-implementación)
    - [Requisitos previos](#requisitos-previos)
    - [Configuración](#configuración)
    - [Instalación](#instalación)
    - [Ejecución](#ejecución)
4. [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 📖 Descripción del Proyecto
El sistema monitoriza el tráfico de red de un servidor web en tiempo real utilizando una arquitectura híbrida:
* **Detección de Anomalías:** Un **Autoencoder** identifica comportamientos inusuales basándose en el error de reconstrucción (detección no supervisada).
* **Clasificación de Amenazas:** Un modelo de **Random Forest** clasifica el tipo específico de ataque (DoS, Fuerza Bruta, Inyecciones SQL).
* **Respuesta Activa:** Integración *Human-in-the-loop* mediante la **API de Telegram**, permitiendo al administrador validar alertas y mitigar ataques en tiempo real bloqueando las IPs atacantes.

---

## 📂 Estructura del Repositorio
```text
AnomaliasTFG/
├── code/               # Scripts .py utilizados para la ejecucion y recapitulación de datos del proyecto
├── data/
    ├── csv/            # Ficheros csv descargados de \cite{CICIDS2017}. No se encuentran subidos al repositorio por falta de espacio     
    ├── df/             # Dataframes utilizados para la implementación del proyecto. Datos preprocesados y analizados    
    ├── model/          # Distintos modelos, tanto redes neuronales, algoritmos de clasificación,  label encoders, scalers, etc.
    ├── umbral/         # Datos capturados para la determinación del umbral definido en (\ref{subsec: determinacion_umbral_deteccion_anomalias})
├── .env                # Fichero de configuración. Importante crearlo y asignar propiedades en función del usuario que utilice la herramienta
├── requirements.txt    # Registro de dependencias del sistema
└── README.md           # Documentación técnica principal
```

---

## 🚀 Despliegue e Implementación

Para poner en marcha el sistema, se han de seguir los pasos detallados a continuación. Debemos contar con privilegios de administrador en el sistema donde se ejecutará la monitorización.

### Requisitos previos
* **Python 3.9+:** Entorno de ejecución recomendado.
* **Privilegios de red:** Dado que el sistema realiza una captura en tiempo real (`nfstream`), es necesario ejecutar el script con permisos elevados (`sudo` en entornos Linux).
* **Bot de Telegram:** Se ha de crear un bot de Telegram a través de [@BotFather](https://t.me/botfather) para obtener la variable de configuracion `BOT_TOKEN`, al igual que `CHAT_ID` que se obtiene interactuando con [@userinfobot](https://t.me/userinfobot).

### Configuración
El proyecto utiliza un archivo `.env` para gestionar las variables sensibles y de configuración. En la raíz del repositorio, se ha de crear un fichero `.env` con las siguientes propiedades:

```bash
# Ruta de clonacion de este repositorio
REPOSITORY_PATH=ruta_donde_tengas_clonado_este_repositorio

# Ruta donde se tenga un diccionario de constraseñas, para la simulacion de ataques
ROCKYOU_PATH=ruta_diccionario

# Configuración de Telegram
TELEGRAM_TOKEN=tu_token_aqui
CHAT_ID=tu_chat_id_aqui

# Otras configuraciones
CSV_CLASS_CAPTURED_PATH=${REPOSITORY_PATH}/data/df/df_malign_captured_cleaned.csv
```

### Instalación
Para preparar el entorno de ejecución, se ha de clonar el repositorio en nuestro sistema, e instalar las librerias utilizadas en el proyecto:

```bash
# Clonar el repositorio
git clone [https://github.com/rnoguer22/AnomaliasTFG.git](https://github.com/rnoguer22/AnomaliasTFG.git)
cd AnomaliasTFG

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución
```bash
# Ejecutar el script con permisos de administrador (sudo), ya que se necesita acceso a la tarjeta de red del sistema para analizar el trafico
sudo python code/stream_detection.py 
```

Esto monitoriza todo el trafico entrante al servidor web, el cual se ha de montar previamente en el sistema en el puerto 80. Para ello, se deben utilizar los archivos `/html/index.php` y `/html/style.css`, los cuales simulan la web de una tienda ficticia con vulnerabilidades criticas, como ataques de fuerza bruta, inyecciones sql, etc. De esta manera estariamos analizando flujos de red y detectando si son potencialmente maliciosos.

---

## 🛠 Tecnologías Utilizadas

El ecosistema técnico del proyecto se basa en herramientas de alto rendimiento diseñadas para el procesamiento de datos y la ciberseguridad:

* **Redes:** `nfstream` es la herramienta base utilizada para la captura, procesado y extracción de características de los flujos de red en tiempo real.
* **Inteligencia Artificial:** * `TensorFlow/Keras`: Empleado para el desarrollo y entrenamiento del **Autoencoder** (detección no supervisada).
    * `Scikit-learn`: Utilizado para la implementación, optimización y evaluación del **Random Forest** (clasificación supervisada).
* **Notificaciones:** `Telegram API` para la creación del canal de comunicación interactivo, permitiendo la gestión de alertas en tiempo real, enviadas a traves de la libreria `requests` de Python.
* **Seguridad:** `iptables` integrado como mecanismo de respuesta activa para la mitigación automática de ataques mediante el bloqueo de IPs en el firewall del servidor.

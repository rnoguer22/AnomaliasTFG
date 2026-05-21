import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv("data/unsw_nb15/UNSW_NB15_testing-set.csv")

def plot_label_distribution(df):
    # Configurar el estilo visual
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    conteo = df['attack_cat'].value_counts()
    print(conteo)
    
    # Crear el gráfico de conteo (barras)
    # Ordenamos las barras de mayor a menor para una mejor lectura
    ax = sns.countplot(
        data=df, 
        x='attack_cat', 
        order=df['attack_cat'].value_counts().index,
        palette='viridis'
    )
    
    # Añadir títulos y etiquetas
    plt.title('Distribución de Clases en el Dataset de Ataques', fontsize=15)
    plt.xlabel('Categoría de Ataque', fontsize=12)
    plt.ylabel('Cantidad de Muestras', fontsize=12)
    plt.xticks(rotation=45) # Rotamos las etiquetas para que no se pisen
    
    # Añadir el número total encima de cada barra
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=10)

    plt.tight_layout()
    plt.show()

# Uso:
plot_label_distribution(df)
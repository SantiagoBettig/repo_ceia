import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import entropy

# Función para graficar histogramas
def plot_histograma(data, column, figsize=(6, 3), bins=15, kde=True, mvd=True, snk=False):
    skewness = (data[column]).skew()
    kurtosis = (data[column]).kurt()
    media = (data[column]).mean()
    var = (data[column]).var()
    std = (data[column]).std()
    plt.figure(figsize=figsize)
    plt.grid(axis='y')
    sns.histplot(data[column], bins=bins, kde=kde)
    if snk:
        plt.figtext(0.7, 0.8, f'Asimetría: {skewness:.2f}', fontsize=10, color='blue')
        plt.figtext(0.715, 0.73, f'Curtosis: {kurtosis:.2f}', fontsize=10, color='blue')
    plt.axvline(media, color='red', linestyle='--', label='Media')
    if mvd:
        plt.figtext(0.15, 0.8, f'Media: {media:.2f}', fontsize=10, color='red')
        plt.figtext(0.15, 0.73, f'Var: {var:.2f}', fontsize=10, color='red')
        plt.figtext(0.15, 0.66, f'Std: {std:.2f}', fontsize=10, color='red')
    plt.title(f'Variable: {column}')
    plt.xlabel(f'{column}')
    plt.ylabel('Frecuencia')
    plt.show()
    
    return

# Función para graficar tortas filtrando por columna (O categoría)
def graficos_torta_por_categoria(df, col_filtro, col_distribucion, n_first=5, cols=3):
    valores = df[col_filtro].dropna().unique()
    filas = -(-len(valores) // cols)  # Techo de la división

    fig, axes = plt.subplots(filas, cols, figsize=(cols * 5, filas * 5))
    axes = axes.flatten()

    for i, valor in enumerate(valores):
        df_filtrado = df[df[col_filtro] == valor]
        conteo = df_filtrado[col_distribucion].value_counts()[:n_first]

        axes[i].pie(
            conteo,
            labels=conteo.index,
            autopct='%1.1f%%',
            startangle=90
        )
        axes[i].set_title(f'{col_filtro} = {valor}', fontsize=13, fontweight='bold')

    # Ocultar gráficos vacíos si la grilla no es exacta
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Distribución de "{col_distribucion}" por "{col_filtro}"', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def heatmap_contingencia(df, col1, col2, n=10, figsize=(10, 6)):
    """
    Muestra un heatmap con la cantidad de cruces entre dos columnas categóricas,
    usando solo los n valores más frecuentes de cada columna.

    Parámetros:
    - df:      DataFrame de pandas
    - col1:    Columna para el eje Y
    - col2:    Columna para el eje X
    - n:       Cantidad de valores más frecuentes a considerar (default: 10)
    - figsize: Tamaño del gráfico (default: (10, 6))
    """
    top_col1 = df[col1].value_counts().head(n).index
    top_col2 = df[col2].value_counts().head(n).index

    df_filtrado = df[df[col1].isin(top_col1) & df[col2].isin(top_col2)]

    tabla = pd.crosstab(df_filtrado[col1], df_filtrado[col2])

    plt.figure(figsize=figsize)
    sns.heatmap(
        tabla,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=0.5
    )
    plt.title(f'Top {n} cruces entre "{col1}" y "{col2}"', fontsize=13, fontweight='bold')
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.tight_layout()
    plt.show()

def boxplots_log(df, columnas, cols=3, figsize=(15, 5)):
    filas = -(-len(columnas) // cols)
    colores = cm.coolwarm(np.linspace(0, 1, len(columnas)))

    fig, axes = plt.subplots(filas, cols, figsize=(figsize[0], figsize[1] * filas))
    axes = axes.flatten()

    for i, col in enumerate(columnas):
        # Extraer columna, quitar ceros y resetear índice
        serie = df[col][df[col] != 0].dropna().reset_index(drop=True)

        sns.boxplot(y=serie, ax=axes[i], color=colores[i])
        axes[i].set_yscale('log')
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Valor (log)')
        axes[i].set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Boxplots en escala logarítmica', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
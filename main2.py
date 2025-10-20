import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando dados
print("Carregando dados...")
df_gapminder = pd.read_csv('gapminder_full.csv')

# Visualizando dados iniciais e sua amostragem
print("\nVisualizando dados iniciais e sua amostragem:")
print(df_gapminder.head())

# Visualizando metadados
print("\nVisualizando metadados:")
df_gapminder.info()

# Remoção duplicados
print("\nRemovendo duplicados...")
df_gapminder.drop_duplicates(inplace=True)

# Remoção de nulos
print("\nRemovendo nulos...")
df_gapminder.dropna(inplace=True)

# Obtendo as estatisticas descritivas
print("\nEstatísticas descritivas:")
print(df_gapminder.describe())

# Total de linhas
print("\nTotal de linhas:")
len(df_gapminder)

# Linhas únicas
print("\nLinhas únicas:")
df_gapminder['country'].nunique()

#Transformação de tipologia
print("\nTransformação de tipologia...")
df_gapminder['pop'] = df_gapminder['pop'].astype('int64')

#Verificando informações do dataframe
print("\nVerificando informações do dataframe...")
df_gapminder.info()

#Verificando resultados
print("\nVerificando resultados...")
print(df_gapminder.head())

#Remover espaços em branco
print("\nRemovendo espaços em branco dos nomes dos países, se houver...")
df_gapminder['country'] = df_gapminder['country'].str.strip()

# Observação de resultado
print("\nObservação de resultado...")
print(df_gapminder.head())
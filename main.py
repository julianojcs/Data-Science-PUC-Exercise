import pandas as pd
import numpy as np # Para operações numéricas, se necessário
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações para melhor visualização dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# Carregar o dataset
try:
    df_gapminder = pd.read_csv('gapminder_full.csv')
    print("Dataset 'gapminder_full.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: O arquivo 'gapminder_full.csv' não foi encontrado. Certifique-se de que ele está no mesmo diretório do notebook.")
    exit() # Encerrar a execução se o arquivo não for encontrado

print(df_gapminder.head())

# Amostra de 5 linhas aleatórias
print("Amostra de 5 linhas aleatórias:")
print(df_gapminder.sample(5))

# Informações do DataFrame
print("Informações do DataFrame:")
df_gapminder.info()

# Estatísticas descritivas
print("Estatísticas descritivas:")
print(df_gapminder.describe())

# Verificar o número de registros únicos por país e ano para entender a granularidade dos dados
print(f"\nNúmero total de registros: {len(df_gapminder)}")
print(f"Número de países únicos: {df_gapminder['country'].nunique()}")
print(f"Intervalo de anos: {df_gapminder['year'].min()} - {df_gapminder['year'].max()}")

# Lista registros únicos
print("Registros únicos:")
df_gapminder["country"].nunique()
df_gapminder["continent"].nunique()

# Trocar o tipo de "pop" de float pra int64
df_gapminder["pop"] = df_gapminder["pop"].astype("int64")

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(df_gapminder.isnull().sum())

# Remover espaços em branco dos nomes dos países, se houver
print("\nRemovendo espaços em branco dos nomes dos países, se houver...")
df_gapminder["country"] = df_gapminder['country'].str.strip()

# Adicionar coluna com o logaritmo do PIB per capita
print("\nAdicionando coluna com o logaritmo do PIB per capita...")
df_gapminder["log_gdpPercap"] = np.log(df_gapminder["gdpPercap"]).round(0)
print(df_gapminder.head())

# Verificando a existência de valores ausentes após a limpeza
print("\nVerificando valores ausentes após a limpeza:")
print(df_gapminder.isnull().sum())

# Excluir linhas com dados ausentes
print("\nExcluindo linhas com dados ausentes...")
df_gapminder.dropna().copy()

# Eliminar dados duplicados
print("\nEliminando dados duplicados...")
df_gapminder.drop_duplicates(inplace=True)

print(df_gapminder.head())

# plt.figure(figsize=(10, 6))
# sns.histplot(df_gapminder['lifeExp'], kde=True, bins=30, color='skyblue')
# plt.title('Distribuição da Expectativa de Vida Global (1952-2007)')
# plt.xlabel('Expectativa de Vida (anos)')
# plt.ylabel('Frequência')
# plt.show()

# # Gráfico de expectativa de vida média por país (top 20)
# avg_life_by_country = df_gapminder.groupby('country')['lifeExp'].mean().sort_values(ascending=False)
# plt.figure(figsize=(12, 8))
# avg_life_by_country.head(20).plot(kind='bar', color='lightgreen')
# plt.title('Expectativa de Vida Média por País (Top 20)')
# plt.xlabel('País')
# plt.ylabel('Expectativa de Vida Média (anos)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Gráfico de expectativa de vida ao longo dos anos para países selecionados
# selected_countries = ['Brazil', 'United States', 'China', 'India', 'Germany']
# df_selected = df_gapminder[df_gapminder['country'].isin(selected_countries)]
# plt.figure(figsize=(12, 8))
# sns.lineplot(data=df_selected, x='year', y='lifeExp', hue='country', marker='o')
# plt.title('Expectativa de Vida ao Longo dos Anos para Países Selecionados')
# plt.xlabel('Ano')
# plt.ylabel('Expectativa de Vida (anos)')
# plt.legend(title='País')
# plt.grid(True)
# plt.show()


def plot_correlation_scatter(df, x_indicator, y_indicator, size_indicator=None, title=None,
                             xlabel=None, ylabel=None, log_scale_x=False, log_scale_y=False):
    """
    Cria um gráfico de dispersão para mostrar a correlação entre dois indicadores.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        x_indicator (str): Nome da coluna para o eixo X.
        y_indicator (str): Nome da coluna para o eixo Y.
        size_indicator (str, optional): Nome da coluna para definir o tamanho dos pontos (ex: 'pop').
        title (str, optional): Título do gráfico.
        xlabel (str, optional): Rótulo do eixo X.
        ylabel (str, optional): Rótulo do eixo Y.
        log_scale_x (bool): Se True, o eixo X será em escala logarítmica.
        log_scale_y (bool): Se True, o eixo Y será em escala logarítmica.
    """
    plt.figure(figsize=(12, 7))

    if size_indicator:
        # Normaliza o tamanho dos pontos para visualização (evita pontos muito grandes)
        sizes = df[size_indicator] / df[size_indicator].max() * 500
        sns.scatterplot(x=x_indicator, y=y_indicator, size=sizes, hue='continent',
                        data=df, alpha=0.7, sizes=(20, 1000), legend='full')
    else:
        sns.scatterplot(x=x_indicator, y=y_indicator, hue='continent', data=df, alpha=0.7)

    plt.title(title if title else f'Correlação entre {y_indicator} e {x_indicator}')
    plt.xlabel(xlabel if xlabel else x_indicator.replace('gdpPercap', 'PIB per Capita').replace('lifeExp', 'Expectativa de Vida'))
    plt.ylabel(ylabel if ylabel else y_indicator.replace('gdpPercap', 'PIB per Capita').replace('lifeExp', 'Expectativa de Vida'))

    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Filtrar dados para o ano de 2007
df_2007 = df_gapminder[df_gapminder['year'] == 2007]
plot_correlation_scatter(df_2007,
                         x_indicator='pop',
                         y_indicator='lifeExp',
                         title='Expectativa de Vida vs. População por País (2007)',
                         xlabel='População (Escala Logarítmica)',
                         ylabel='Expectativa de Vida (anos)',
                         log_scale_x=True)
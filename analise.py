# ANÁLISE DE QUEIMADAS NO BRASIL (JAN 2016 – JAN 2026)
# Autor: Ronaldo Oliveira
# Objetivo: Aplicar técnicas de análise de dados para identificar padrões, prever riscos e apoiar decisões
# estratégicas na mitigação das queimadas no Brasil.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# pip install pyarrow fastparquet  # Para o uso do formato parquet indiretamente
# pip install scikit-learn

# Desabilitando avisos.
warnings.filterwarnings('ignore')

pd.set_option('display.width', None)  # Mostra toda a coluna sem os cortes com as reticências.

# Lendo o arquivo Parquet e salvando em Datafame novo.
df = pd.read_parquet(r'C:\Users\onlys\PycharmProjects\analise_queimadas_brasil\data_parquet\base_dados_parquet.parquet')

# insert(posição, nome_coluna, valores)
df.insert(0, 'ID', range(1, len(df) + 1))

print('\nANÁLISE PÓS TRATAMENTO DOS DADOS:', '\n')

# Conferindo resultado do dataframe em parquet.
print("VISÃO DO INICIO DA BASE DE DADOS EM .PARQUET: \n", df.head(5), '\n\n')
print("VISÃO DO FINAL DA BASE DE DADOS EM .PARQUET: \n", df.tail(5), '\n\n')

# Importante para a otimização. Cada dígito equivale a 10x a memória usada.
memoria_utilizada = df.memory_usage(deep=True).sum()
print(f"TOTAL DE MEMÓRIA UTILIZADA: {memoria_utilizada / 1_000_000_000} GB \n")

# =====================
# ANÁLISE ESTATÍSTICAS
# =====================

print("\n--------------\n ESTATÍSTICAS\n--------------\n")

print('INFORMAÇÕES GERAIS DOS DADOS DA AMOSTRA PÓS TRATAMENTO (TIPAGEM): ')
print(df.info(), '\n')

print('\nDESCREVENDO DE FORMA GERAL MEDIDAS CENTRAIS E DE DISPERSÃO DE TODOS OS CAMPOS: \n', df.describe(), '\n')  # Descrevendo de forma geral as estatísticas.

print('\nDESCREVENDO DE FORMA GERAL MEDIDAS CENTRAIS E DE DISPERSÃO DOS VALORE NUMÉRICOS: \n', df[['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP']].describe(), '\n\n')

print('SEPARATRIZES: \n')
print('MÍNIMO DE DiaSemChuva:', df['DiaSemChuva'].min())  # Minimo Quartil 0.
print('MÁXIMO DE DiaSemChuva: ', df['DiaSemChuva'].max())  # Maximo. Quartil 100.
print('QUARTIS DiaSemChuva: ', '\n', df['DiaSemChuva'].quantile([0.25, 0.5, 0.75]), '\n')  # Para visualizar os quartis. A posição 0 é como o minimo. A posição 100 é o maximo.
print('MÉDIA DE DiaSemChuva: ', df['DiaSemChuva'].mean())  # Média
print('MEDIANA DE DiaSemChuva: ', df['DiaSemChuva'].median())  # Mediana
print('VARIÂNCIA DE DiaSemChuva: ', df['DiaSemChuva'].var())   # Variância
print('DESVIO PADRÃO DE  DiaSemChuva: ', df['DiaSemChuva'].std())  # Desvio padrão
print('MODA DE DiaSemChuva: ', df['DiaSemChuva'].mode()[0], '\n')  # Moda. O [0] é para mostra apenas um registro. Não todos.

print('MÍNIMO DE Precipitacao:', df['Precipitacao'].min())
print('MÁXIMO DE Precipitacao: ', df['Precipitacao'].max())
print('QUARTIS Precipitacao: ', '\n', df['Precipitacao'].quantile([0.25, 0.5, 0.75]), '\n')
print('MÉDIA DE Precipitacao: ', df['Precipitacao'].mean())
print('MEDIANA DE Precipitacao: ', df['Precipitacao'].median())
print('VARIÂNCIA DE Precipitacao: ', df['Precipitacao'].var())
print('DESVIO PADRÃO DE  Precipitacao: ', df['Precipitacao'].std())
print('MODA DE Precipitacao: ', df['Precipitacao'].mode()[0], '\n')

print('MÍNIMO DE RiscoFogo:', df['RiscoFogo'].min())
print('MÁXIMO DE RiscoFogo: ', df['RiscoFogo'].max())
print('QUARTIS RiscoFogo: ', '\n', df['RiscoFogo'].quantile([0.25, 0.5, 0.75]), '\n')
print('MÉDIA DE RiscoFogo: ', df['RiscoFogo'].mean())
print('MEDIANA DE RiscoFogo: ', df['RiscoFogo'].median())
print('VARIÂNCIA DE RiscoFogo: ', df['RiscoFogo'].var())
print('DESVIO PADRÃO DE  RiscoFogo: ', df['RiscoFogo'].std())
print('MODA DE RiscoFogo: ', df['RiscoFogo'].mode()[0], '\n')

print('MÍNIMO DE FRP:', df['FRP'].min())
print('MÁXIMO DE FRP: ', df['FRP'].max())
print('QUARTIS DE FRP: ', '\n', df['FRP'].quantile([0.25, 0.5, 0.75]), '\n')
print('MÉDIA DE FRP: ', df['FRP'].mean())
print('MEDIANA DE FRP: ', df['FRP'].median())
print('VARIÂNCIA DE FRP: ', df['FRP'].var())
print('DESVIO PADRÃO DE  FRP: ', df['FRP'].std())
print('MODA DE FRP: ', df['FRP'].mode()[0], '\n\n\n')

# Índices de queimadas
print("\nÍNDICES DE QUEIMADAS: \n")

# Soma do total do campo FRP.
total_FRP = df['FRP'].sum()
print("TOTAL QUEIMADAS ACUMULADO NO PERIODO DE (jan 2016 a jan 2026) TODOS OS BIOMAS:", total_FRP)

# FRP Soma por Ano ordenado do maior para o menor resultado.
FRP_por_ano = df[["Ano", "FRP"]].groupby("Ano", as_index=True).sum().sort_values(by="FRP", ascending=False)
print("\nTOTAL QUEIMADAS POR ANO: \n\n", FRP_por_ano)

# FRP Soma por Bioma e Estado.
FRP_por_bioma_estado = df[["Bioma", "Estado", "FRP"]].groupby(["Bioma", "Estado"]).sum()
print("\nTOTAL QUEIMADAS POR BIOMA E ESTADO: \n", FRP_por_bioma_estado)

# FRP Soma por Bioma ordenado do maior para o menor resultado.
FRP_por_bioma = df[["Bioma", "FRP"]].groupby("Bioma", as_index=True).sum().sort_values(by="FRP", ascending=False)
print("\nTOTAL QUEIMADAS POR BIOMA: \n", FRP_por_bioma)

# Precipitacao Soma por Estado ordenado do maior para o menor resultado.
FRP_por_estado = df[["Estado", "FRP"]].groupby("Estado", as_index=True).sum().sort_values(by="FRP", ascending=False)
print("\nTOTAL QUEIMADAS POR ESTADO: \n", FRP_por_estado.head(30))

# FRP Soma por Estado ordenado do maior para o menor resultado os 20 maiores em ocorrência de FRP.
FRP_por_municipio = df[["Municipio", "FRP"]].groupby("Municipio", as_index=True).sum().sort_values(by="FRP", ascending=False)
print("\nTOTAL QUEIMADAS POR MUNICIPIO, 20 MAIORES: \n", FRP_por_municipio.head(20), '\n')

# Índices de Precipitação.
print("\nÍNDICES DE PRECIPITAÇÃO: \n")

# Soma do total do campo Precipitacao.
total_Precipitacao = df['Precipitacao'].sum()
print("TOTAL Precipitacao ACUMULADO NO PERIODO DE (jan 2016 a jan 2026) TODOS OS BIOMAS:", total_Precipitacao)

# Precipitacao Soma por Bioma ordenado do maior para o menor resultado.
Precipitacao_por_bioma = df[["Bioma", "Precipitacao"]].groupby("Bioma", as_index=True).sum().sort_values(by="Precipitacao", ascending=False)
print("\nTOTAL Precipitacao POR BIOMA: \n", Precipitacao_por_bioma)

# Precipitacao Soma por Estado ordenado do maior para o menor resultado.
Precipitacao_por_estado = df[["Estado", "Precipitacao"]].groupby("Estado", as_index=True).sum().sort_values(by="Precipitacao", ascending=False)
print("\nTOTAL Precipitacao POR ESTADO: \n", Precipitacao_por_estado.head(30))

# Precipitacao Soma por Estado ordenado do maior para o menor resultado os 20 maiores em ocorrencia de FRP.
Precipitacao_por_municipio = df[["Municipio", "Precipitacao"]].groupby("Municipio", as_index=True).sum().sort_values(by="Precipitacao", ascending=False)
print("\nTOTAL Precipitacao POR MUNICIPIO, 20 MAIORES: \n", Precipitacao_por_municipio.head(20), '\n')

# Precipitacao Soma por Bioma e Estado.
Precipitacao_por_bioma_estado = df[["Bioma", "Estado", "Precipitacao"]].groupby(["Bioma", "Estado"]).sum()
print("\nTOTAL Precipitacao POR BIOMA E ESTADO: \n", Precipitacao_por_bioma_estado)

# Índices de Risco Fogo.
print("\nÍNDICES DE RISCO FOGO: \n")

# RiscoFogo Soma por Bioma ordenado do maior para o menor resultado.
RiscoFogo_por_bioma = df[["Bioma", "RiscoFogo"]].groupby("Bioma", as_index=True).sum().sort_values(by="RiscoFogo", ascending=False)
print("TOTAL RiscoFogo POR BIOMA: \n", RiscoFogo_por_bioma)

# RiscoFogo Soma por Estado ordenado do maior para o menor resultado.
RiscoFogo_por_estado = df[["Estado", "RiscoFogo"]].groupby("Estado", as_index=True).sum().sort_values(by="RiscoFogo", ascending=False)
print("\nTOTAL RiscoFogo POR ESTADO: \n", RiscoFogo_por_estado.head(30))

# RiscoFogo Soma por Estado ordenado do maior para o menor resultado os 20 maiores em ocorrencia de FRP.
RiscoFogo_por_municipio = df[["Municipio", "RiscoFogo"]].groupby("Municipio", as_index=True).sum().sort_values(by="RiscoFogo", ascending=False)
print("\nTOTAL RiscoFogo POR MUNICIPIO, 20 MAIORES: \n", RiscoFogo_por_municipio.head(20), '\n')

# RiscoFogo Soma por Bioma e Estado.
RiscoFogo_por_bioma_estado = df[["Bioma", "Estado", "RiscoFogo"]].groupby(["Bioma", "Estado"]).sum()
print("\nTOTAL RiscoFogo POR BIOMA E ESTADO: \n\n", RiscoFogo_por_bioma_estado)

# Índices de dias sem chuva.
print("\nÍNDICES DE DIAS SEM CHUVA: \n")

# Soma do total do campo DiaSemChuva Pelos indicadores regionais.
total_DiaSemChuva = df['DiaSemChuva'].sum()
print("TOTAL DiaSemChuva ACUMULADO NO PERIODO DE (jan 2016 a jan 2026):", total_DiaSemChuva)

# DiaSemChuva Soma por Bioma ordenado do maior para o menor resultado.
DiaSemChuva_por_bioma = df[["Bioma", "DiaSemChuva"]].groupby("Bioma", as_index=True).sum().sort_values(by="DiaSemChuva", ascending=False)
print("\nTOTAL DiaSemChuva POR BIOMA: \n", DiaSemChuva_por_bioma)

# DiaSemChuva Soma por Estado ordenado do maior para o menor resultado.
DiaSemChuva_por_estado = df[["Estado", "DiaSemChuva"]].groupby("Estado", as_index=True).sum().sort_values(by="DiaSemChuva", ascending=False)
print("\nTOTAL DiaSemChuva POR ESTADO: \n", DiaSemChuva_por_estado.head(30))

# DiaSemChuva Soma por Estado ordenado do maior para o menor resultado os 20 maiores em ocorrencia de FRP.
DiaSemChuva_por_municipio = df[["Municipio", "DiaSemChuva"]].groupby("Municipio", as_index=True).sum().sort_values(by="DiaSemChuva", ascending=False)
print("\nTOTAL DiaSemChuva POR MUNICIPIO, 20 MAIORES: \n", DiaSemChuva_por_municipio.head(20))

# DiaSemChuva Soma por Bioma e Estado.
DiaSemChuva_por_bioma_estado = df[["Bioma", "Estado", "DiaSemChuva"]].groupby(["Bioma", "Estado"]).sum()
print("\nTOTAL DiaSemChuva POR BIOMA E ESTADO: \n\n", DiaSemChuva_por_bioma_estado)

print('\nCORRELAÇÕES: \n')
# Guia de correlações Pearson e Sperman:
# Muito fraca:   Valor de correlação: 0.00 a 0.19
# Fraca:         Valor de correlação: 0.20 a 0.39
# Moderada:      Valor de correlação: 0.40 a 0.59
# Forte:         Valor de correlação: 0.60 a 0.79
# Muito forte:   Valor de correlação: 0.80 a 1.00

print('CORRELAÇÃO ENTRE DiaSemChuva, Precipitacao, RiscoFogo e FRP: ')

# pearson_corr = df[['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP']].corr()
pearson_corr = df[['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP']].corr(method='pearson')
spearman_corr = df[['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP']].corr(method='spearman')
kendall_corr = df[['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP']].corr(method="kendall")

print('\nCORRELAÇÃO DE PERSON: \n', pearson_corr)
print('\nCORRELAÇÃO DE SPEARMAN: \n', spearman_corr)
print('\nCORRELAÇÃO DE KENDALL: \n', kendall_corr)

print('\nCORRELAÇÃO ENTRE DiaSemChuva, Precipitacao, RiscoFogo e FRP COM _MinMaxScaler: \n')

# pearson_corr = df[['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP']].corr()
pearson_corr_MinMaxScaler = df[['DiaSemChuva_MinMaxScaler', 'Precipitacao_MinMaxScaler', 'RiscoFogo_MinMaxScaler', 'FRP_MinMaxScaler']].corr(method='pearson')
spearman_corr_MinMaxScaler = df[['DiaSemChuva_MinMaxScaler', 'Precipitacao_MinMaxScaler', 'RiscoFogo_MinMaxScaler', 'FRP_MinMaxScaler']].corr(method='spearman')
kendall_corr_MinMaxScaler = df[['DiaSemChuva_MinMaxScaler', 'Precipitacao_MinMaxScaler', 'RiscoFogo_MinMaxScaler', 'FRP_MinMaxScaler']].corr(method="kendall")

print('\nCORRELAÇÃO DE PERSON _MinMaxScaler: \n', pearson_corr_MinMaxScaler)
print('\nCORRELAÇÃO DE SPEARMAN _MinMaxScaler: \n', spearman_corr_MinMaxScaler)
print('\nCORRELAÇÃO DE KENDALL _MinMaxScaler: \n', kendall_corr_MinMaxScaler)

# =======================================
# ANÁLISE EXPLORATÓRIA COM GRÁFICOS (EDA).
# =======================================

print('\nPLOTS DOS GRÁFICOS: \n')

# As amostras não são distribuídas de maneira uniforme. Alguns Biomas recebem mais leituras que outros.
# Assim como a distribuição anual.

# Biomas com maior quantidades leituras pelos satélites.
contagem_bioma = df["Bioma"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(
    x=contagem_bioma.index,
    y=contagem_bioma.values,
    palette="viridis"
)
plt.title("Distribuição de Registros de Queimadas por Bioma", fontsize=16, weight="bold")
plt.xlabel("Bioma", fontsize=12)
plt.ylabel("Quantidade de Registros", fontsize=12)
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# Satélite com maior quantidades leituras históricas.
contagem_bioma = df["Satelite"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(
    x=contagem_bioma.index,
    y=contagem_bioma.values,
    palette="viridis"
)
plt.title("Distribuição de Registros de Leitura por Satélite", fontsize=16, weight="bold")
plt.xlabel("Satélite", fontsize=12)
plt.ylabel("Quantidade de Registros", fontsize=12)
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# Ano com maior quantidades leituras pelos satélites. Gráfico de pizza.
contagem = df["Ano"].value_counts().sort_index()
plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(
    contagem,
    labels=contagem.index,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    pctdistance=0.85,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
# Criando círculo central para efeito "donut"
centre_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)
plt.setp(autotexts, size=12, weight="bold", color="black")  # Melhorando texto
plt.setp(texts, size=12)
plt.title("Distribuição Percentual de Registros por Ano", fontsize=16, weight="bold")  # Título
plt.axis('equal')
plt.show()

# Gráfico de dispersão. cmap= é para múltiplas cores
plt.hexbin(df['DiaSemChuva'], df['RiscoFogo'], gridsize=30, cmap='Blues')
plt.colorbar(label='Contagem dentro do bin')  # Para a barra lateral de contagem.
plt.xlabel('DiaSemChuva')
plt.ylabel('RiscoFogo')
plt.title('Dispersão de DiaSemChuva X RiscoFogo')
plt.show()

# Anos com maior registro de focos de queimadas.
contagem_bioma = df["Ano"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(
    x=contagem_bioma.index,
    y=contagem_bioma.values,
    palette="viridis"
)
plt.title("Distribuição de Registros de Queimadas por Ano", fontsize=16, weight="bold")
plt.xlabel("Ano", fontsize=12)
plt.ylabel("Quantidade de Registros", fontsize=12)
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# Correlação
correlacao = df[['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP']].corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlação entre Variáveis de Queimadas", fontsize=14, weight="bold")
plt.show()

# Importante para a otimização. Cada dígito equivale a 10x a memória usada.
memoria_utilizada = df.memory_usage(deep=True).sum()
print(f"\nTOTAL DE MEMÓRIA UTILIZADA: {memoria_utilizada / 1_000_000_000} GB \n")
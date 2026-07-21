# ANÁLISE DE QUEIMADAS NO BRASIL (JAN 2016 – JAN 2026)
# Autor: Ronaldo Oliveira
# Objetivo: Aplicar técnicas de análise de dados para identificar padrões, prever riscos e apoiar decisões
# estratégicas na mitigação das queimadas no Brasil.

import pandas as pd
import warnings
import os
from sklearn.preprocessing import MinMaxScaler
# pip install pyarrow fastparquet  # Para o uso do formato parquet indiretamente
# pip install scikit-learn

# Desabilitando avisos.
warnings.filterwarnings('ignore')

# Caminho local dos dados distribuídos em múltiplas tabelas tipo .csv.
caminho_dados = r"C:\Users\onlys\PycharmProjects\analise_queimadas_brasil\data_csv_arquivo_unico\1_janeiro_a_ 31_dezembro_de_2016_a_2026.csv"

# Dataframe vazio.
base_dados_concatenada = pd.DataFrame()

base_dados = pd.read_csv(caminho_dados, index_col=False)

# Tratando os dados numéricos para salvar no formato parquet.
base_dados['DiaSemChuva'] = pd.to_numeric(base_dados['DiaSemChuva'], errors='coerce')
base_dados['Precipitacao'] = pd.to_numeric(base_dados['Precipitacao'], errors='coerce')
base_dados['RiscoFogo'] = pd.to_numeric(base_dados['RiscoFogo'], errors='coerce')
base_dados['FRP'] = pd.to_numeric(base_dados['FRP'], errors='coerce')
base_dados['Latitude'] = pd.to_numeric(base_dados['Latitude'], errors='coerce')
base_dados['Longitude'] = pd.to_numeric(base_dados['Longitude'], errors='coerce')

# Tratando o índice automáticos.
base_dados = base_dados.drop(columns=['Unnamed: 0'], errors='ignore')

# =========================
# DADOS INICIAIS DO DATASET
# =========================
print("---------------------------\n DADOS INICIAIS DO DATASET\n---------------------------")

pd.set_option('display.width', None)  # Mostra toda a coluna sem os cortes com as reticências.

# Importante para a otimização. Cada dígito equivale a 10x a memória usada.
memoria_utilizada = base_dados.memory_usage(deep=True).sum()
print(f"TOTAL DE MEMÓRIA UTILIZADA: {memoria_utilizada / 1_000_000_000} GB \n")

# Verificando dados iniciais, primeiros e últimos registros.
print("\nVISÃO DO INICIO DA BASE DE DADOS IMPORTADA SEM TRATAMENTOS: \n", base_dados.head(3).to_string(), '\n\n')

# Verificando qtd de linhas e colunas iniciais da base da amostra.
print('TOTAL DE REGISTROS INICIAIS:', base_dados.shape[0], '\nTOTAL DE COLUNAS INICIAIS: ', base_dados.shape[1])

# Verificando nulos.
print('TOTAL DOS DADOS NULOS: ', base_dados.isnull().sum().sum(), '\n')  # Para conferir o total dos dados nulos.
print('TOTAL DOS DADOS DUPLICADOS: ', base_dados.duplicated().sum(), '\n')
print("TOTAL VALORES NULOS POR COLUNAS: \n", base_dados.isnull().sum(), '\n')
print("TOTAL EM PORCENTAGEM (%) DE VALORES DE DADOS NULOS: \n", base_dados.isnull().mean() * 100, '\n\n')

# Verificando tipos de dados.
print('TIPAGEM DOS DADOS DA AMOSTRA:\n', base_dados.dtypes, '\n')

# Verificando informações como tipos de dados e valores nulos com o função .info().
print('INFORMAÇÕES GERAIS DOS DADOS DA AMOSTRA: \n')
print(base_dados.info(), '\n')

# ===============================
# INICIANDO O TRATAMENTO DE DADOS
# ===============================
print("---------------------------------\n INICIANDO O TRATAMENTO DE DADOS\n---------------------------------\n")

# # Removendo dados não necessários no próprio dataframe. As políticas são feitas por estado e não por coordenadas.
# # base_dados.drop('Latitude', axis=1, inplace=True)
# # base_dados.drop('Longitude', axis=1, inplace=True)

# Removendo campo "Pais" do dataframe. Todos os dados são sobre o Brasil, campo irrelevante.
base_dados.drop('Pais', axis=1, inplace=True)

# Conversão do campo DataHora tipo object para datetime.
base_dados["DataHora"] = pd.to_datetime(base_dados["DataHora"], errors="coerce")

# print("VISÃO DO INICIO DA BASE DE DADOS COM FILTRO DE COLUNAS RETIRADAS, (Pais, Latitude E Longitude):\n", base_dados.head(3).to_string(), '\n\n')

# Verificando qtd de linhas e colunas pós-remoção de colunas irrelevantes.
print('TOTAL PÓS REMOÇÃO DE COLUNAS IRRELEVANTES: ', base_dados.shape[1], '\n')

# Tratando valores nulos (ausentes).
base_dados = base_dados.dropna(thresh=4)  # Define um limite mínimo de valores válidos (não nulos) que a linha deve ter para ser mantida

# Conferindo se há valores negativos nas medições.
print('QUANTIDADE DE VALORES NEGATIVOS EM DiaSemChuva: ', (base_dados['DiaSemChuva'] < 0).sum())
print('QUANTIDADE DE VALORES NEGATIVOS EM Precipitacao: ', (base_dados['Precipitacao'] < 0).sum())
print('QUANTIDADE DE VALORES NEGATIVOS EM RiscoFogo: ', (base_dados['RiscoFogo'] < 0).sum())
print('QUANTIDADE DE VALORES NEGATIVOS EM FRP: ', (base_dados['FRP'] < 0).sum(), '\n\n')

# Tratando os valores negativos para zero, porque não pode chover negativo ou risco de queimada menor que zero. Outliers inferiores.
base_dados['DiaSemChuva'] = base_dados['DiaSemChuva'].clip(lower=0)
base_dados['Precipitacao'] = base_dados['Precipitacao'].clip(lower=0)
base_dados['RiscoFogo'] = base_dados['RiscoFogo'].clip(lower=0)
base_dados['FRP'] = base_dados['FRP'].clip(lower=0)

# Tratando os valores nulos para zero.
base_dados['DiaSemChuva'] = base_dados['DiaSemChuva'].fillna(0)  # Substitui valores nulos por 0 nesse caso. A menor quantidade de dias com chuva só pode ser zero.
base_dados['Precipitacao'] = base_dados['Precipitacao'].fillna(0)  # Substitui valores nulos por 0 nesse caso. A menor quantidade de Precipitacao só pode ser zero.
base_dados['RiscoFogo'] = base_dados['RiscoFogo'].fillna(0)  # Substitui valores nulos por 0 nesse caso. A menor quantidade de Risco de Fogo só pode ser zero.
base_dados['Bioma'] = base_dados['Bioma'].fillna('Indefinido')  # Substitui valores nulos por 'Indefinido'.

# base_dados['FRP'] = base_dados['FRP'].fillna(0)  # Substitui valores nulos por 0 nesse caso. Para testar o efeitos nas estatísticas.
base_dados['FRP'] = base_dados['FRP'].fillna(base_dados['FRP'].mean())  # Substitui os valores nulos pela média para não afetar os cálculos de medidas centrais.

# Conferir se ainda tem registros negativos e salvando em variáveis.
print("AINDA HÁ DADOS NEGATIVOS NA AMOSTRA?")
qtd_negativos_DiaSemChuva = (base_dados['DiaSemChuva'] < 0).sum()
qtd_negativos_Precipitacao = (base_dados['Precipitacao'] < 0).sum()
qtd_negativos_RiscoFogo = (base_dados['RiscoFogo'] < 0).sum()
qtd_negativos_FRP = (base_dados['FRP'] < 0).sum()
print('QUANTIDADE DE VALORES NEGATIVOS EM DiaSemChuva: ', qtd_negativos_DiaSemChuva)
print('QUANTIDADE DE VALORES NEGATIVOS EM Precipitacao: ', qtd_negativos_Precipitacao)
print('QUANTIDADE DE VALORES NEGATIVOS EM RiscoFogo: ', qtd_negativos_RiscoFogo)
print('QUANTIDADE DE VALORES NEGATIVOS EM FRP: ', qtd_negativos_FRP, '\n')

# Conferindo se ainda tem registros nulos salvando em variáveis.
print("AINDA HÁ DADOS NULOS NA AMOSTRA? \n", base_dados.isnull().sum(), '\n\n')

# Excluindo duplicatas. Deixar essa etapa sempre por último.
base_dados.drop_duplicates()  # Remove todas as duplicatas.
print('QUANTIDADE DE REGISTROS APÓS REMOÇÃO DAS DUPLICATAS:', base_dados.shape[0], '\n')  # Conferindo a quantidade na remoção das duplicatas. # df.shape[0] Mostra o total de linhas duplicadas.

# Conferindo resultado da substituição.
print("VISÃO DO INICIO DA BASE DE DADOS PÓS SUBSTITUIÇÃO DOS VALORES NULOS, NEGATIVOS, DUPLICATAS E DA MÉDIA DE FRP:\n", base_dados.head(8).to_string(), '\n\n')

# Não faz sentido no contexto retirar outliers superiores. Os inferiores e nulos já foram tratados.
print('VALORES NA BANDA SUPERIOR DA AMOSTRA:')
print('MAIOR VALOR DE DiaSemChuva: ', base_dados['DiaSemChuva'].max())
print('MAIOR VALOR DE Precipitacao: ', base_dados['Precipitacao'].max())
print('MAIOR VALOR DE RiscoFogo: ', base_dados['RiscoFogo'].max())
print('MAIOR VALOR DE FRP: ', base_dados['FRP'].max(), '\n')

# Tratando dados preenchidos errados(Por analise visual).
base_dados = base_dados[base_dados["Bioma"] != "BIOMA"]
base_dados = base_dados[base_dados["Satelite"] != "Satelite"]

# ===========================================
# NORMALIZANDO OS DADOS E FEATURE ENGINEERING
# ===========================================

print("------------------------\n NORMALIZANDO OS DADOS E\n FEATURE ENGINEERING\n------------------------\n")

# Normalizando campos de texto. Definindo o padrão como caixa alta em toda a coluna.
base_dados['Bioma'] = base_dados['Bioma'].str.strip().str.upper()

# Criando os campos ano/mês para as análises temporais.
base_dados["AnoMes"] = base_dados["DataHora"].dt.to_period("M").dt.to_timestamp()
base_dados['Ano'] = base_dados['DataHora'].dt.year

# Normalização - MinMaxScaler. Normalizar os campos e pô-los em ESCALAS numéricas que possam ser comparadas entre si.
# Criando campos para receber esses dados. Assim mantém os originais para tipos diferentes de análises.
scaler = MinMaxScaler()

# Normaliza entre 0 e 1. Para facilitar na estatística.
base_dados['DiaSemChuva_MinMaxScaler'] = scaler.fit_transform(base_dados[['DiaSemChuva']])
base_dados['Precipitacao_MinMaxScaler'] = scaler.fit_transform(base_dados[['Precipitacao']])
base_dados['RiscoFogo_MinMaxScaler'] = scaler.fit_transform(base_dados[['RiscoFogo']])
base_dados['FRP_MinMaxScaler'] = scaler.fit_transform(base_dados[['FRP']])

# Mostrando os dados numéricos com a normalização MinMaxScaler.
print("DADOS NUMÉRICOS EM MINMAXSCALER ENTRE (De 0 a 1):")
print("DiaSemChuva MaxScaler: Min: {:.4}   /  Max: {:.4}   /  Mean: {:.4}   /  Std: {:.4}". format(base_dados['DiaSemChuva_MinMaxScaler'].min(), base_dados['DiaSemChuva_MinMaxScaler'].max(), base_dados['DiaSemChuva_MinMaxScaler'].mean(), base_dados['DiaSemChuva_MinMaxScaler'].std()))
print("Precipitacao MaxScaler: Min: {:.4}   /  Max: {:.4}   /  Mean: {:.4}   /  Std: {:.4}". format(base_dados['Precipitacao_MinMaxScaler'].min(), base_dados['Precipitacao_MinMaxScaler'].max(), base_dados['Precipitacao_MinMaxScaler'].mean(), base_dados['Precipitacao_MinMaxScaler'].std()))
print("RiscoFogo MaxScaler: Min: {:.4}   /  Max: {:.4}   /  Mean: {:.4}   /  Std: {:.4}". format(base_dados['RiscoFogo_MinMaxScaler'].min(), base_dados['RiscoFogo_MinMaxScaler'].max(), base_dados['RiscoFogo_MinMaxScaler'].mean(), base_dados['RiscoFogo_MinMaxScaler'].std()))
print("FRP MaxScaler: Min: {:.4}   /  Max: {:.4}   /  Mean: {:.4}   /  Std: {:.4}". format(base_dados['FRP_MinMaxScaler'].min(), base_dados['FRP_MinMaxScaler'].max(), base_dados['FRP_MinMaxScaler'].mean(), base_dados['FRP_MinMaxScaler'].std()), '\n\n')

# Tratando dados de texto (Estado, Municipio, Bioma). Atribuindo números para a categorizar.
base_dados['Estado_cod'] = base_dados['Estado'].astype('category').cat.codes
base_dados['Municipio_cod'] = base_dados['Municipio'].astype('category').cat.codes
base_dados['Bioma_cod'] = base_dados['Bioma'].astype('category').cat.codes

# =============================================
# SALVANDO EM UM ARQUIVO TIPO PARQUET, NA PASTA
# data_parquet PARA OTIMIZADA A ANÁLISE.
# =============================================

# Definindo o caminho da pasta e o nome do arquivo
pasta_destino = 'data_parquet'
nome_arquivo = 'base_dados_parquet.parquet'
caminho_completo = os.path.join(pasta_destino, nome_arquivo)

# Criando a pasta se ela não existir
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Salvando o DataFrame em formato Parquet, 'snappy' é uma compressão rápida e padrão no Parquet.
base_dados.to_parquet(caminho_completo, engine='pyarrow', compression='snappy', index=False)

# Lendo o arquivo Parquet e salvando dem datafame novo.
df = pd.read_parquet(r'C:\Users\onlys\PycharmProjects\analise_queimadas_brasil\data_parquet\base_dados_parquet.parquet')

# insert(posição, nome_coluna, valores)
df.insert(0, 'ID', range(1, len(df) + 1))

# Conferindo resultado do dataframe em Parquet.
print("VISÃO DO INICIO DA BASE DE DADOS PÓS NORMALIZAÇÃO, FEATURE E CONVERSÃO PARA .PARQUET: \n", df.head(15), '\n\n')
print("VISÃO DO FINAL DA BASE DE DADOS PÓS NORMALIZAÇÃO, FEATURE E CONVERSÃO PARA .PARQUET: \n", df.tail(15), '\n\n')

print('INFORMAÇÕES GERAIS DOS DADOS DA AMOSTRA PÓS NORMALIZAÇÃO E FEATURE: \n')
print(base_dados.info(), '\n')

# Importante para a otimização. Cada dígito equivale a 10x a memória usada.
memoria_utilizada = base_dados.memory_usage(deep=True).sum()
print(f"TOTAL DE MEMÓRIA UTILIZADA: {memoria_utilizada / 1_000_000_000} GB \n")

print('FINALIZADA A ETAPA DE PREPARAÇÃO DOS DADOS.\n')
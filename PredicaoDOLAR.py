# #### Etapas
#
# - Passo 0 - Decidir o que modelar - OK
# - Passo 1 - Carregar as bibliotecas - OK
# - Passo 2 - Carregar a base de dados - OK
# - Passo 3 - Tratamento dos dados - OK
# - Passo 4 - Construçao dos alvos e variaveis
# - Passo 5 - Divisão da base para treinamento e teste
# - Passo 6 - Treinamento do modelo
# - Passo 7 - Avaliaçao dos Resultados
# - Passo 8 - Otimizaçao com Grid Search
# - Passo 9 - Treinamento com o modelo otimizado
# - Passo 10 - Ensemble simples

# - Passo 1 - Importar as bibliotecas
import datetime as dt
from unittest.mock import inplace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sb
import os.path
yf.pdr_override()
pd.set_option('display.max_columns', None)


# - Passo 2 - Carregar a base de dados
if 1 == 0 and os.path.isfile('dolar.csv') and os.path.isfile('dxy.csv') and os.path.isfile('ibov.csv') :
        print ("File exist")
        dolar = pd.read_csv("dolar.csv")
        dolar.set_index('Date', inplace=True)
        dxy = pd.read_csv("dxy.csv")
        dxy.set_index('Date', inplace=True)
        ibov = pd.read_csv("ibov.csv")
        ibov.set_index('Date', inplace=True)
else:
        inicio = dt.datetime(2003,12,1)
        fim = dt.datetime(2021,2,26)
        fim = dt.datetime(2021,3,26)
        dolar = yf.download("USDBRL=X",start = inicio,end = fim)        #Cotação da taxa de câmbio
        dxy = yf.download("DX-Y.NYB",start = inicio,end = fim)          #Dólar Índice
        ibov = yf.download("^BVSP",start = inicio,end = fim)            #Índice Bovespa
        dolar.to_csv("dolar.csv")
        dxy.to_csv("dxy.csv")
        ibov.to_csv("ibov.csv")

# print(dolar.count())
# print(dxy.count())
# print(ibov.count())

# print("Dolar => ", dolar.loc['2020-12-30':'2021-01-04'])
# print("DXY => ", dxy.loc['2020-12-30':'2021-01-04']['Close'])
# print("IBOV => ", ibov.loc['2020-12-30':'2021-01-04']['Close'])


# - Passo 3 - Tratamento dos dados
dolar.drop(columns=['Adj Close', 'Volume'], inplace=True)

data = dolar
data['dxy'] = dxy['Close']
data['ibov'] = ibov['Close']


# print(data.loc['2020-12-30':'2021-01-04'])
# print(data.isna().sum())

data = data.interpolate(method='linear', limit_direction='forward')

# print(data.isna().sum())

# data = data.round({"ibov":0})
# print(data.loc['2020-12-01':'2020-12-10'])

# print(data[~data.index.isin(data.index)].count())
# print(data.describe())
# print(dolar.describe())
# print(data.isna().sum())


##- Passo 4 - Construçao dos alvos e variaveis


periodos = 1 # Períodos anteriores

# Calculo do ganho, variação percentual
data["ganhoAtual"] = data["Close"].pct_change(periodos)
data["ganhoFuturo"] = data["ganhoAtual"].shift(-periodos)

# - Alvo a ser modelado
data["comprarVender"] = np.where(data.ganhoFuturo > 0, 1, 0)
# data["comprarVender"] = np.where(data['ganhoFuturo'] > data["ganhoAtual"].describe()[6]/2 , 1
#                            , np.where(data['ganhoFuturo'] < data["ganhoAtual"].describe()[4]/2, -1, 0))


# Boxplot do Ganho Futuro pelo Alvo definido, compra ou venda.
data.boxplot(by ="comprarVender", column =["ganhoFuturo"], grid = True, figsize = (8,6));
plt.title("% Ganho por Compra ou Venda")
plt.suptitle("")
plt.ylabel("% Ganho")
plt.xlabel("(0) Venda e (1) Compra")
plt.show()

# data.boxplot(column="variacao", figsize=(12,7))
# plt.show()


# raise SystemExit(0)

# Ibovespa dolarizado
data["ibovDol"] = data['ibov']/data['Close']

# Variaveis do dólar
data["dolar_OC"] = (data["Close"] - data["Open"])/data["Close"]    # Quantos pontos de variacao da Abertura para o Fechamento do dia
data["dolar_CH"] = (data["Close"] - data["High"])/data["Close"]    # Quantos pontos de variacao do Fechamento para o maior preco do dia - Pavio
data["dolar_CL"] = (data["Close"] - data["Low"])/data["Close"]    # Quantos pontos de variacao do Fechamento para o menor preco do dia - Pavio
data["dolar_spread"] = (data["High"] - data["Low"])/data["Close"]  # Quantos pontos de variacao do High ao Low - Spread

data['dolar_trend_curta'] = np.where(data['Close'] > data["Close"].rolling(3).mean(), '1', '0')     # Valor de fechamento maior que a media, 1, menor, 0
data['dolar_trend_media'] = np.where(data['Close'] > data["Close"].rolling(5).mean(), '1', '0')    # Valor de fechamento maior que a media, 1, menor, 0
data['dolar_trend_longa'] = np.where(data['Close'] > data["Close"].rolling(10).mean(), '1', '0')   # Valor de fechamento maior que a media, 1, menor, 0

data['dolar_std_5'] = data["Close"].rolling(5).std()
data['dolar_std_10'] = data["Close"].rolling(10).std()

# Variaveis do dólar índice - IDX
data['dxy_trend_curta'] = np.where(data['dxy'] > data["dxy"].rolling(3).mean(), '1', '0')   # Valor de fechamento maior que a media, 1, menor, 0
data['dxy_trend_media'] = np.where(data['dxy'] > data["dxy"].rolling(5).mean(), '1', '0')  # Valor de fechamento maior que a media, 1, menor, 0
data['dxy_trend_longa'] = np.where(data['dxy'] > data["dxy"].rolling(10).mean(), '1', '0') # Valor de fechamento maior que a media, 1, menor, 0


# Variaveis do ibovespa
data['ibov_trend_curta'] = np.where(data['ibov'] > data["ibov"].rolling(3).mean(), '1', '0')        # Valor de fechamento maior que a media, 1, menor, 0
data['ibov_trend_media'] = np.where(data['ibov'] > data["ibov"].rolling(5).mean(), '1', '0')       # Valor de fechamento maior que a media, 1, menor, 0
data['ibov_trend_longa'] = np.where(data['ibov'] > data["ibov"].rolling(10).mean(), '1', '0')      # Valor de fechamento maior que a media, 1, menor, 0

# Variaveis do ibovespa dolarizado
data['ibovDol_trend_curta'] = np.where(data['ibovDol'] > data["ibovDol"].rolling(3).mean(), '1', '0')        # Valor de fechamento maior que a media, 1, menor, 0
data['ibovDol_trend_media'] = np.where(data['ibovDol'] > data["ibovDol"].rolling(5).mean(), '1', '0')       # Valor de fechamento maior que a media, 1, menor, 0
data['ibovDol_trend_longa'] = np.where(data['ibovDol'] > data["ibovDol"].rolling(10).mean(), '1', '0')      # Valor de fechamento maior que a media, 1, menor, 0


data_modelo = data.dropna()

# print(data_modelo.head())



data_modelo.to_excel("base.xlsx")



# - Passo 5 - Divisão da base para treinamento e teste

data_inicial = data_modelo.index[0]   #Data inicial: 2003-12-12
data_final = data_modelo.index[-1]     #Data final: 2021-02-25

print("data inicial: ", data_inicial, " ####### data final: ", data_final)

# Data para treinamento - 2003-12-12 a 2014-12-31
inicio_train = data_inicial
fim_train = "2012-12-31"

# Data para teste - 2015-01-01 a 2021-02-25
inicio_teste = "2013-01-01"
fim_teste = data_final

# Dividindo os dados entre treinamento e teste
data_modelo_train = data_modelo[inicio_train : fim_train]
data_modelo_teste = data_modelo[inicio_teste : fim_teste]

train_x = data_modelo_train.iloc[:, 10:data_modelo.shape[1]]    # Os varios X do meu treinamento
train_y = data_modelo_train.comprarVender                       # O Y do meu treinamento

teste_x = data_modelo_teste.iloc[:, 10:data_modelo.shape[1]]    # Os varios X do meu teste
teste_y = data_modelo_teste.comprarVender                       # O Y do meu teste

# print(train_x.head())
# print(train_y.head())


# - Passo 6 - Treinamento do modelo
from sklearn.tree import DecisionTreeClassifier

arvore = DecisionTreeClassifier()

arvore.fit(train_x, train_y)

teste_pred = arvore.predict(teste_x)


# - Passo 7 - Avaliaçao dos Resultados
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(teste_y, teste_pred))
print("___________________________________________________________")
print(classification_report(teste_y, teste_pred))

# Acuracia
import sklearn.metrics as metrics
print("Acuracia: ", round(metrics.accuracy_score(teste_y, teste_pred,3)*100))

# plt.figure(figsize = (12,5))
# plt.hist(data["variacao"], bins = 50
#         , alpha = 0.45
#         , bottom = 10
#         , histtype = "stepfilled"
#         , color = "darkgreen"
#         , edgecolor = "none"
#         , label = "Retornos");
#
# plt.legend()
# plt.title("Distribuiçao Retornos");
# plt.show()


# Criacao do alvo
# data["RetornoBin"] = np.where(data['retornoFuturo'] > data["variacao"].describe()[6]/2 , 1
#                            , np.where(data['retornoFuturo'] < data["variacao"].describe()[4]/2, -1, 0))



# print(data[data.retornoAtual == data.retornoAtual.min()])

# plt.figure(figsize = (12,5))
# plt.hist(data["ResultadoBin"], bins = 5
#         , alpha = 0.45
#         , histtype = "stepfilled"
#         , color = "darkgreen"
#         , edgecolor = "none"
#         , label = "Retornos");
#
# plt.legend()
# plt.title("Distribuiçao Bin");







# raise SystemExit(0)

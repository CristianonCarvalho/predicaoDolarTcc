# #### Etapas
#
# - Passo 0 - Decidir o que modelar
# - Passo 1 - Carregar as bibliotecas
# - Passo 2 - Carregar a base de dados
# - Passo 3 - Tratamento dos dados
# - Passo 4 - Construçao dos alvos e variaveis
# - Passo 5 - Preparaçao da base para treinamento e teste
# - Passo 6 - Treinamento do modelo
# - Passo 7 - Avaliaçao dos Resultados
# - Passo 8 - Otimizaçao com Grid Search
# - Passo 9 - Treinamento com o modelo otimizado
# - Passo 10 - Ensemble simples


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import yfinance as yf
import investpy as inv

yf.pdr_override()

# // YF não possui VOLUME
dolar = web.get_data_yahoo("USDBRL=X", start='2003-12-01', end='2021-02-27')['Close']
dxy = web.get_data_yahoo("DX-Y.NYB", start='2003-12-01', end='2021-02-27')['Close'] #https://br.tradingview.com/symbols/TVC-DXY/
ibov = web.get_data_yahoo("^BVSP", start='2003-12-01', end='2021-02-27')['Close']


# Calcular IBOV dolarizado
# carteira["IBOV_DOLARIZADO"] = (carteira["IBOV"] / carteira["DOLAR"])
# print(dolar.head())
# print(dxy.head())
# print(ibov.head())


data = pd.DataFrame(dolar)
data.rename(columns={1:'USDBRL'}, inplace=True)

data['dxy'] = dxy
data['ibov'] = ibov
data['ibovDol'] = ibov/dolar


print(data)
print(data.describe())




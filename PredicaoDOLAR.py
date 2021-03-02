# #### Etapas
#
# - Passo 0 - Decidir o que modelar - OK
# - Passo 1 - Carregar as bibliotecas - OK
# - Passo 2 - Carregar a base de dados - OK
# - Passo 3 - Tratamento dos dados - OK
# - Passo 4 - Construçao dos alvos e variaveis
# - Passo 5 - Preparaçao da base para treinamento e teste
# - Passo 6 - Treinamento do modelo
# - Passo 7 - Avaliaçao dos Resultados
# - Passo 8 - Otimizaçao com Grid Search
# - Passo 9 - Treinamento com o modelo otimizado
# - Passo 10 - Ensemble simples

# - Passo 1 - Carregar as bibliotecas
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
pd.set_option('display.max_columns', None)


# - Passo 2 - Carregar a base de dados
start = dt.datetime(2003,12,1)
end = dt.datetime(2021,2,27)
dolar = yf.download("USDBRL=X",start = start,end = end)['Close']
dxy = yf.download("DX-Y.NYB",start = start,end = end)['Close']
ibov = yf.download("^BVSP",start = start,end = end)['Close']

# - Passo 3 - Tratamento dos dados
data = pd.DataFrame(dolar)
data.rename(columns={'Close':'USDBRL'}, inplace=True)
data['dxy'] = dxy
data['ibov'] = ibov


##- Passo 4 - Construçao dos alvos e variaveis
# Ibovespa dolarizado

periodos = 1 # Períodos anteriores

# lag de X periodos
data["USDBRL_lag"] = data['USDBRL'].shift(-periodos)
data["Retorno"] = data["USDBRL"].pct_change(periodos)
data["Resultado"] = data["Retorno"].shift(-periodos)
data['ibovDol'] = ibov/dolar

# Variação do dólar
data["Spread_USDBRL"] = data["USDBRL"] - data["USDBRL"].shift(1)
data["Spread_USDBRL"] = data["Spread_USDBRL"].shift(-periodos)

# Variação do dólar índice
data["Spread_dxy"] = data["dxy"] - data["dxy"].shift(1)
data["Spread_dxy"] = data["Spread_dxy"].shift(-periodos)

# Variação do ibovespa
data["Spread_ibov"] = data["ibov"] - data["ibov"].shift(1)
data["Spread_ibov"] = data["Spread_ibov"].shift(-periodos)



print(data)
print(data["Retorno"].describe())


plt.figure(figsize = (12,5))
plt.hist(data["Retorno"], bins = 20
        , alpha = 0.25
        , histtype = "stepfilled"
        , color = "darkgreen"
        , edgecolor = "none"
        , label = "Retornos");

plt.legend()
plt.title("Distribuiçao Retornos");

plt.show()


# raise SystemExit(0)

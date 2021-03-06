# Código Fonte para o trabalho de Tópicos Especiais em Econometria
# Autor: Gustavo Alovisi
# Nro: 00243669

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import sqlite3
import statsmodels.tsa.stattools as smtsa
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.stats.diagnostic as smstats
from scipy.stats import chi2
import statsmodels.graphics.gofplots as smgof
import statsmodels.stats.stattools as smtools

jurosmedmensal = pd.read_csv('http://api.bcb.gov.br/dados/serie/bcdata.sgs.25433/dados?formato=csv',
                  sep=';', encoding='utf-8', decimal=',')
txinadimplencia = pd.read_csv('http://api.bcb.gov.br/dados/serie/bcdata.sgs.21082/dados?formato=csv',
                                sep=';', encoding='utf-8', decimal=',')


jurosmedmensal.head()
txinadimplencia.head()


## criação de um dataframe com todos os dados pertinentes
txinadimplencia.columns = ['data1', 'txinadimp']
jurosmedmensal.columns = ['data2', 'juros']
dataframe = [txinadimplencia, jurosmedmensal]
dataframe = pd.concat(dataframe, 1)
dataframe = dataframe.drop('data2', 1)
dataframe.head()

## criação de um banco de dados para nosso trabalho via SQLite
database = sqlite3.connect('dados.db')
c = database.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS variaveis (data text, txjuros real, inadimplencia real)''')

#c.executemany('INSERT into variaveis VALUES (?,?,?)', dataframe)
dataframe.to_sql("variaveis", database, if_exists='replace')

## carregar do DB
econvars = pd.read_sql_query("SELECT * FROM variaveis;", database)
econvars.head()


## análise de nossos dados
plt.subplot(121)
grafJuros = plt.plot(econvars.data1, econvars.juros)
plt.title('Média da taxa de Juros Mensal')
plt.xlabel("tempo")
plt.ylabel("juros ao mês (%)")

plt.subplot(122)
grafInadimplencia = plt.plot(econvars.data1, econvars.txinadimp)
plt.title("Tx de Inadimplência Mensal")
plt.xlabel("tempo")
plt.ylabel("inadimplencia ao mês (%)")

plt.show()

grafCorr = plt.scatter(econvars.txinadimp, econvars.juros)
plt.show()

np.corrcoef(econvars.txinadimp, econvars.juros)


adf = smtsa.adfuller(econvars.txinadimp, regression='ctt')  # 'ctt' para checar a estacionariedade em tendência constante, linear e quadrática
adfoutput_inad = pd.Series(adf[0:4], index=["Estatística do Teste:", "P-valor", "# lags", "# observacoes"])
for key, value in adf[4].items():
        adfoutput_inad['Critical Value ({})'.format(key)] = value
print(adfoutput_inad)


adf = smtsa.adfuller(econvars.juros, regression='ctt')  # 'ctt' para checar a estacionariedade em tendência constante, linear e quadrática
adfoutput_juros = pd.Series(adf[0:4], index=["Estatística do Teste:", "P-valor", "# lags", "# observacoes"])
for key, value in adf[4].items():
        adfoutput_juros['Critical Value ({})'.format(key)] = value
print(adfoutput_juros)


acf = tsaplots.plot_acf(econvars.txinadimp)
acf = tsaplots.plot_acf(econvars.juros)


diff_inadimp = np.diff(econvars.txinadimp)
acf = tsaplots.plot_acf(diff_inadimp, alpha=0.05, title="Autocorrelação da 1a diferença da Taxa de Inadimplência")

diff_diff_inadimp = np.diff(diff_inadimp)
acf = tsaplots.plot_acf(diff_diff_inadimp, alpha=0.05, title="Autocorrelação da 2a diferença da Taxa de Inadimplência")

diff_diff_inadimp = pd.Series(diff_diff_inadimp)
grafdiff_diff_inadimp = plt.plot(diff_diff_inadimp)
plt.show()
diff_diff_inadimp.head()


logdiffdiff = np.log(econvars.txinadimp).diff().diff().dropna()
logdiffdiff = logdiffdiff[13:]
#logdiffdiff.head()
graflogdiffdiff = plt.plot(logdiffdiff)
plt.show()

## Plots de autocorrelação e diff, testes ADF
acf = tsaplots.plot_acf(logdiffdiff, alpha=0.05,  title="Autocorrelação da 2a diferença da logaritmica de Taxa de Inadimplência")

adf_inadimp = smtsa.adfuller(logdiffdiff, regression='ctt')  # 'ctt' para checar a estacionariedade em tendência constante, linear e quadrática
adfout_diff_inad = pd.Series(adf[0:4], index=["Estatística do Teste:", "P-valor", "# lags", "# observacoes"])
for key, value in adf_inadimp[4].items():
        adfout_diff_inad['Critical Value ({})'.format(key)] = value
print(adfout_diff_inad)

diff_juros = np.diff(econvars.juros)
acf = tsaplots.plot_acf(diff_juros, alpha=0.05,  title="Autocorrelação da 1a diferença da Taxa de Juros")

diff_diff_juros = np.diff(diff_juros)
acf = tsaplots.plot_acf(diff_diff_juros, alpha= 0.05,  title="Autocorrelação da 1a diferença da Taxa de Juros")

logdiffdiffjuros = np.log(econvars.juros).diff(14).diff().dropna()  # dif(14) devido a uma aparente sazonalidade em nossa série
acf = tsaplots.plot_acf(logdiffdiffjuros, alpha= 0.05,  title="Autocorrelação da 1a diferença da Taxa de Juros")

graflogdiffdiffjuros = plt.plot(logdiffdiffjuros)
plt.show()

adf_juros = smtsa.adfuller(logdiffdiffjuros, regression='ctt')  #'ctt' para checar a estacionariedade em tendência constante, linear e quadrática
adfout_diff_juros = pd.Series(adf[0:4], index=["Estatística do Teste:", "P-valor", "# lags", "# observacoes"])
for key, value in adf_juros[4].items():
        adfout_diff_juros['Critical Value ({})'.format(key)] = value
print(adfout_diff_juros)

##Ajustamento da regressão por MQO
result_reg = sm.OLS(logdiffdiff, logdiffdiffjuros, missing='drop').fit()
print(result_reg.summary())

## Implementação de um teste de Breusch-Pagan para heterocedasticidade assim como em stasmodels het_breuschpagan (estava dando um erro)
y = np.asarray(result_reg.resid**2)
x = np.asarray(logdiffdiffjuros)
resultadoBP = sm.OLS(y, x).fit()
fval = resultadoBP.fvalue
fpval = resultadoBP.f_pvalue
lm = 71*resultadoBP.rsquared
lmtest = chi2.sf(lm, 70)
print("P-valor do teste:", fpval)
print("Teste LM:", lmtest)

## QQ plot da normalidade dos resíduos
qqplot = smgof.qqplot(result_reg.resid, line = 'q')
plt.show()

## Teste de JB para a normalidade dos resíduos
jarque_bera = smtools.jarque_bera(result_reg.resid)
print("P-valor do teste: ", jarque_bera[1])
print("Skewness estimada: ", jarque_bera[2])
print("Kurtose estimada: ", jarque_bera[3])

## Teste de ljung-box para a autocrrelação dos resíduos
ljung = smstats.acorr_ljungbox(result_reg.resid)
y=0
for x in ljung[1]:
    y += x
resul_lbox = y/len(ljung[1])
print("P-valor do teste:", resul_lbox)
# print(ljung[1])


## Teste de Cointegração de Johansen
coint_test = smtsa.coint(econvars.juros, econvars.txinadimp)
print("P-valor do teste de Cointegração de Johansen: ", coint_test[1])

## Salvar o output em arquivo de texto
arquivo = open("saidasrelatorio.txt", "w") ##abrindo para leitura/gravação e criando se não existir
arquivo.write("Resumo do Relatório Econométrico  \n")
arquivo.write("\nTeste ADF para a Taxa de Inadimplencia:  \n")
arquivo.write(str(adfoutput_inad))
arquivo.write("\n\nTeste ADF para a Taxa de Juros: \n")
arquivo.write(str(adfoutput_juros))
arquivo.write("\n\nTeste ADF para a Taxa de Indimplencia Diferenciada:\n")
arquivo.write(str(adfout_diff_inad))
arquivo.write("\n\nTeste ADF para a Taxa de Juros Diferenciada:\n")
arquivo.write(str(adfout_diff_juros))
arquivo.write("\n\nResultado da Regressao por MQO:\n")
arquivo.write("\n" + str(result_reg.summary()))
arquivo.write("\n\nResultado do Teste de Ljung-Box para autocorrelação dos resíduos:\n")
arquivo.write("Pvalor do teste:" + str(resul_lbox))
arquivo.write("\n\nTeste de Heterocedasticidade de Breusch-Pagan:\n")
arquivo.write("Pvalor do teste:" + str(fpval))
arquivo.write("\n\nTeste de Jarque-Bera da Normalidade dos Resíduos:\n")
arquivo.write("Pvalor do teste:" + str(jarque_bera[1]))
arquivo.write("\n\nTeste de Cointegração de Johansen:\n")
arquivo.write("Pvalor do teste:" + str(coint_test[1]))
arquivo.close() ##fechando o arquivo
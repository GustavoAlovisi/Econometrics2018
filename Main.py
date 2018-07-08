import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

txjuros = pd.read_csv('http://api.bcb.gov.br/dados/serie/bcdata.sgs.25433/dados?formato=csv)', sep=
";", encoding='utf-8', decimal =',')

pib = pd.read_csv('http://api.bcb.gov.br/dados/serie/bcdata.sgs.22099/dados?formato=csv',
                  sep = ';', encoding = 'utf-8', decimal = ',')
pib.head()




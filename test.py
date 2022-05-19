import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import statsmodels functions
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis


# Import custom functions
from neer import neer
from grangers import grangers_causation_matrix
from cointegration import cointegration_test, adjust
from adFuller import adfuller_test
from invertTransformation import invert_transformation
from forecastAcc import forecast_accuracy

#--------------------------------------------------------------------
# Create all the columns for our VAR variables
dfI = pd.DataFrame(columns = ['date','oil','OG','M3','NEER','CPI'])


# Use dates from Jan 2010 to 2020
date = pd.read_csv('dates.csv')
dfI['date'] = date

# Oil prices copied from yahoo finance
oil = pd.read_csv('oil.csv')
dfI['oil'] = oil

# Calculate Output Gap from the industrial PPI using HP filter
ppi = pd.read_csv('ppi.csv')
cycle, trend = hpfilter(ppi,1600)
dfI['OG'] = trend

# M3 money supply from FastBull website
m3 = pd.read_csv('m3.csv')
dfI['M3'] = m3

# We calculate the Nominal effective exchange rate 0.4 dollar 0.6 euro
# From investing.com
USXR = pd.read_csv('USXR.csv')
EUXR = pd.read_csv('EUXR.csv')
NEER = []
for i in range(len(df)):
    temp = neer(USXR['Price'].iloc[0],EUXR['Price'].iloc[0],
    USXR['Price'].iloc[i],USXR['Price'].iloc[i])
    NEER.append(temp)

dfI['NEER'] = NEER


# CPI from HCP website

cpi = pd.read_csv('cpi.csv')
dfI['CPI'] = cpi['CPI']

# Set the dates as index and inverse for oldest to newest plots
dfI=dfI.set_index('date')
dfI = dfI.iloc[::-1]


print(f'    Data collected', "\n   ", '-'*47)
print(dfI)

df = pd.DataFrame(columns = ['date','oil','OG','M3','NEER','CPI'])
df['date'] = dfI['date'].iloc[1:len(dfI)]

print(df)
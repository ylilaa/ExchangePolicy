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
ipi = pd.read_csv('ipi.csv')
cycle, trend = hpfilter(ipi['ipi'],14400)
dfI['OG'] = trend

# M3 money supply from FastBull website
m3 = pd.read_csv('m3.csv')
dfI['M3'] = m3

# We calculate the Nominal effective exchange rate 0.4 dollar 0.6 euro
# From investing.com
USXR = pd.read_csv('USXR.csv')
EUXR = pd.read_csv('EUXR.csv')
NEER = []
for i in range(len(dfI)):
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

df = dfI.pct_change()
df['OG'] = dfI['OG']
df.drop('Jan-10', axis=0, inplace=True)

print(f'    Data in percentage change', "\n   ", '-'*47)
print(df)


# ---------------------------------------------------------------------
# VAR Model calculations

# Plot
# df.plot(subplots=True, layout=(3,2),use_index=True)
print(f'    Granger Test results', "\n   ", '-'*47)
print(grangers_causation_matrix(df, variables = df.columns)) 

print(f'    Cointegration Test results', "\n   ", '-'*47)
cointegration_test(df)

nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]

# Check size
print(df_train.shape)  # (119, 8)
print(df_test.shape)  # (4, 8)

# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# 1st difference
df_differenced = df_train.diff().dropna()

# ADF Test on each column of 1st Differences Dataframe
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# Second Differencing
df_differenced = df_differenced.diff().dropna()

# ADF Test on each column of 2nd Differences Dataframe
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

model = VAR(df_differenced)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

x = model.select_order(maxlags=12)
x.summary()

model_fitted = model.fit(4)
model_fitted.summary()

print(f'    Durbin Watson Test results', "\n   ", '-'*47)
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(adjust(col), ':', round(val, 2))

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
df_forecast
df_results = invert_transformation(df_train, df_forecast, second_diff=True)
df_results.loc[:, ['oil_forecast', 'OG_forecast', 'M3_forecast', 'NEER_forecast',
                   'CPI_forecast']]


fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=3, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()


print('Forecast Accuracy of: Oil')
accuracy_prod = forecast_accuracy(df_results['oil_forecast'].values, df_test['oil'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: OG')
accuracy_prod = forecast_accuracy(df_results['OG_forecast'].values, df_test['OG'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: M3')
accuracy_prod = forecast_accuracy(df_results['M3_forecast'].values, df_test['M3'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: NEER')
accuracy_prod = forecast_accuracy(df_results['NEER_forecast'].values, df_test['NEER'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: CPI')
accuracy_prod = forecast_accuracy(df_results['CPI_forecast'].values, df_test['CPI'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))


irf = IRAnalysis(model_fitted)

irf.plot(impulse='oil', response='OG')
irf.plot(impulse='oil', response='CPI')
irf.plot(impulse='OG', response='M3')
irf.plot(impulse='oil', response='NEER')
irf.plot(impulse='M3', response='CPI')
irf.plot(impulse='NEER', response='OG')
irf.plot(impulse='NEER', response='M3')
irf.plot(impulse='NEER', response='CPI')
irf.plot(impulse='CPI', response='M3')
irf.plot(impulse='CPI', response='NEER')
plt.show()





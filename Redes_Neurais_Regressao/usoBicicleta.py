# %% Importando as libs necessárias 

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% Novas importações ao longo do projeto

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# %% Consumindo o csv e vizualizando os dados

bike_df = pd.read_csv('bike-sharing-daily.csv')

print(bike_df)
print(bike_df.info())

sns.heatmap(bike_df.isnull());

# %% Limpeza de Dados - pt1

bike_df = bike_df.drop(labels=['instant'], axis = 1)
print(bike_df.head())

# %% Limpeza de Dados pt2

bike_df = bike_df.drop(labels=['casual','registered'], axis = 1)
print(bike_df.head())

# %% Limpeza de Dados pt3

bike_df.dteday = pd.to_datetime(bike_df.dteday, format ='%m/%d/%Y')
print(bike_df.head())

# %% Limpeza de Dados pt4

bike_df.index = pd.DatetimeIndex(bike_df.dteday)
bike_df = bike_df.drop(labels=['dteday'], axis = 1)
print(bike_df.head())

# %% Vizualização dos dados pt1

bike_df['cnt'].asfreq('W').plot(linewidth = 3)
plt.title('Bike usage per week')
plt.xlabel('Week')
plt.ylabel('Bike rental');

# %% Visualização dos dados pt2
bike_df['cnt'].asfreq('M').plot(linewidth = 3)
plt.title('Bike usage per month')
plt.xlabel('Month')
plt.ylabel('Bike rental');

# %% Visualização dos dados pt3
bike_df['cnt'].asfreq('Q').plot(linewidth = 3)
plt.title('Bike usage per quarter')
plt.xlabel('Quarter')
plt.ylabel('Bike rental');

# %% Visualização dos dados pt4

sns.pairplot(bike_df) #Gera gráficos com todos os atributos do dataset

# %% Visualização dos dados pt5

x_numerical = bike_df[['temp', 'hum', 'windspeed', 'cnt']]
sns.pairplot(x_numerical);

# %% Visualização dos dados pt6

#Mostra um mapa de calor relacionando os atributos do dataset
sns.heatmap(x_numerical.corr(), annot = True); 

# %% Tratamento das bases de dados pt1

x_cat = bike_df[['season','yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]
print(x_cat.head())

# %% Tratamento das bases de dados pt2

onehotencoder = OneHotEncoder()
x_cat = onehotencoder.fit_transform(x_cat).toarray()

# %% Tratamento das bases de dados pt3

x_cat = pd.DataFrame(x_cat)
print(x_cat.head())

# %% Tratamento das bases de dados pt4

x_numerical = x_numerical.reset_index()
x_all = pd.concat([x_cat,x_numerical], axis = 1)

print(x_all.head())

# %% Tratamento das bases de dados pt5

x_all = x_all.drop (labels=['dteday'], axis=1)
print (x_all.head())

# %% Tratamento das bases de dados pt6 - Definindo as variáveis

x = x_all.iloc[:, :-1].values
y = x_all.iloc[:, -1:].values

print(x)
print(y)

# %% Tratamento das bases de dados pt7

scaler = MinMaxScaler()
y = scaler.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# %% Treinamento do Modelo pt1 - Estruturando a rede neural

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units =100, activation='relu', input_shape=(35,)))
model.add(tf.keras.layers.Dense(units =100, activation='relu'))
model.add(tf.keras.layers.Dense(units =100, activation='relu'))
model.add(tf.keras.layers.Dense(units =1, activation='linear'))

model.summary()

# %% Treinamento do Modelo pt2 - Compilando a rede neural

model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = model.fit(x_train, y_train, epochs = 25, batch_size = 50, validation_split=0.2)

# %% Validação do Modelo pt1 - Gráficos

# epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss']);
plt.plot(epochs_hist.history['val_loss']);
plt.title('Model loss progress during training')
plt.xlabel('Epochs')
plt.ylabel('Training and validation loss')
plt.legend(['Training loss', 'Validation loss']);

# %% Validação do Modelo pt2 - Previsões

y_predict = model.predict(x_test)
plt.plot(y_test,y_predict, "^", color='r')
plt.xlabel('Model predictions')
plt.ylabel('True Values');

# %% Validação do Modelo pt3 - Previsões

y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)

plt.plot(y_test_orig,y_predict_orig, "^", color='r')
plt.xlabel('Model predictions')
plt.ylabel('True Values');

# %% Validação do Modelo pt3 - Metrics de Erro

k = x_test.shape[1]
n = len(x_test)

print(k,n)

mae = mean_absolute_error(y_test_orig,y_predict_orig)
mse = mean_squared_error(y_test_orig,y_predict_orig)
rmse = sqrt(mse)
r2 = r2_score(y_test_orig,y_predict_orig)
adj_r2 = 1 - (1-r2) * (n-1) / (n-k-1)

print("MAE: ", mae, "\nMSE: ", mse, "\nRMSE: ", rmse, "\nr2: ", r2, "\nADJ R2: ", adj_r2)
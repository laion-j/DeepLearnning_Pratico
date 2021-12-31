# %% Importando as libs necessárias e consumindo o CSV

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

lucro_df = pd.read_csv('SalesData.csv')
lucro_df.reset_index(drop = True, inplace = True)

#print(lucro_df)

# %% Separando os dados

x_train = lucro_df['Temperature']
y_train = lucro_df['Revenue']

# %% Setando o modelo e colocando para treino

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=10, input_shape = [1]))
model.add(tf.keras.layers.Dense(units=1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss = 'mean_squared_error')

epochs_hist = model.fit(x_train,y_train, epochs=1000)
epochs_hist.history.keys()

#%% Gerando gráfico para vizualizar os resultados

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Épocas')
plt.ylabel('Taxa de erro')
plt.legend(['Taxa de erro']);

# %% Gerando gráfico para vizualizar os resultados

plt.scatter(x_train,y_train, color = 'gray')
plt.plot(x_train, model.predict(x_train), color = 'red')
plt.ylabel('Revenue [$]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand');

#%% Analisando o modelo

model.get_weights()

# %% Utilizando o modelo para prever

temp_c = 5
temp_f = model.predict([temp_c])

print(temp_f)
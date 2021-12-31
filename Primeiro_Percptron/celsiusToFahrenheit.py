# %%

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

temperature_df = pd.read_csv('Celsius-to-Fahrenheit.csv')
temperature_df.reset_index(drop = True, inplace = True)

# print(temperature_df.info())
# print(temperature_df.describe())

# sns.scatterplot(temperature_df['Celsius'], temperature_df['Fahrenheit']);

x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']

# print(x_train)
# print(y_train)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.8), loss = 'mean_squared_error')

epochs_hist = model.fit(x_train,y_train, epochs=1000)
epochs_hist.history.keys()

# %%

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Épocas')
plt.ylabel('Taxa de erro')
plt.legend(['Taxa de erro']);

# %%

model.get_weights()

# %%
temp_c = 10
temp_f = model.predict([temp_c])

print(temp_f)
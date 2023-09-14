
import pandas as pd

dataset = pd.read_csv("kassal_product.csv")

x = dataset.drop(columns=['barcode', 'Price in NOK', 'nutrient', 'ingridients', 'store', 'image', 'src'])

print(x)

y = dataset['barcode']

print(y)
print(y[0])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')


print("model print \n",model)
print(x_train)
print(y_train)

model.fit(x_train, y_train, epochs=1000)

model.evaluate(x_test, y_test)
print(y)
print(y[0])

"""jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0"""


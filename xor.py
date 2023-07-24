import numpy as np
from tensorflow import keras


X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
y = np.array([[0],[1],[1],[0]], "float32")


model = keras.models.Sequential()


model.add(keras.layers.Dense(8, input_dim=2, activation='relu'))  # hidden layer
model.add(keras.layers.Dense(1, activation='sigmoid'))  # output layer


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])


model.fit(X, y, epochs=1000)


print(model.predict(X).round())

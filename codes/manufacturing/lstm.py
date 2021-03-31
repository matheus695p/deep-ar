import keras
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
from datetime import datetime, timedelta
from src.module import (index_date, always_number, downcast_dtypes,
                        lstm_preparation, delete_negatives, training_history,
                        lstm_metric_evaluation)
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore", category=DeprecationWarning)

# seteo de matplotlib
mpl.rcParams['figure.figsize'] = (20, 12)
mpl.rcParams['axes.grid'] = False

# lectura de los datos en .txt
df = pd.read_csv('data/manufacturing_r30.csv', parse_dates=True)
# df = pd.read_csv('data/manufacturing.csv', parse_dates=True)
# agrupar los datos en 10 dias
# df = grouping_df(df, days=10, arg="sum")
df.reset_index(drop=True, inplace=True)
df = always_number(df)
fecha_inicial = df["fecha"].iloc[0][0: 10]
fecha_final = df["fecha"].iloc[-1][0: 10]
print("Fecha inicial del historial de ventas: ", fecha_inicial)
print("Fecha final del historial de ventas: ", fecha_final)

# buscar indice de finalizacion
end_date = datetime.strptime(df["fecha"].iloc[-1],
                             "%Y-%m-%d %H-%M-%S") - timedelta(days=44)
end_date = end_date.strftime("%Y-%m-%d %H-%M-%S")

# separar los datos
test_index = df[df["fecha"] == end_date].index[0]
# número de timesteps con los que se va a trabajar
timesteps = 30
# a una freuencia de 15 minutos
days_of_prediction = len(df) - test_index
# start_date
start_date = df["fecha"].iloc[0][0: 10]
end_date = df["fecha"].iloc[test_index][0: 10]
# fechas del conjunto de testing
fechas = np.array(df["fecha"].iloc[test_index:])
# fecha como timestamp
df["fecha"] = df["fecha"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H-%M-%S"))
# indexar fecha
df = index_date(df)
# bajar IOPS
df = downcast_dtypes(df)
# nombres
names = list(df.columns)
# filter_names = []
# for name in names:
#     print(name)
#     name = name.replace("rolling_", "")
#     filter_names.append(name)

# datos de training
train_df = df.iloc[0: test_index]
# datos de testing con timesteps hacia atrás
test_df = df.iloc[test_index - timesteps:]
print(train_df.info())

# reset de indices para eliminar las fechas
df.reset_index(drop=True, inplace=True)
# normalizar los datos
sc = MinMaxScaler(feature_range=(0, 1))
# training
train_df = sc.fit_transform(train_df)
# testing
test_df = sc.transform(test_df)
# hacer reshape para las transformaciones de las celdas lstm
x_train, y_train = lstm_preparation(train_df, timesteps=timesteps)
x_test, y_test = lstm_preparation(test_df, timesteps=timesteps)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# definición de hiperparámetros
hyperparameters = {
    "penalization": True,
    "batch_size": 20,
    "epochs": 500,
    "patience": 50,
    "min_delta": 10e-7,
    "optimizer": "adam",
    "lr_factor": 0.75,
    "lr_patience": 25,
    "lr_min": 1E-3,
    "validation_size": 0.2}
# tamaño del lote de entrenamiento
batch_size = hyperparameters["batch_size"]
# épocas de entrenamiento
epochs = hyperparameters["epochs"]
# paciencia del earlystopping
patience = hyperparameters["patience"]
# lo mínimo que tiene que bajar el earlystopping para terminar el train
min_delta = hyperparameters["min_delta"]
# optimizador
optimizer = hyperparameters["optimizer"]
# factor de división learining rate
lr_factor = hyperparameters["lr_factor"]
# paciencia del learning rate
lr_patience = hyperparameters["lr_patience"]
# mínima tasas de aprendizaje que puede llegar a tener
lr_min = hyperparameters["lr_min"]
# porcentaje de los datos a validación
validation_size = hyperparameters["validation_size"]

# modelo
lstm = tf.keras.Sequential()
lstm.add(tf.keras.layers.LSTM(units=256,
                              input_shape=(
                                  np.array(x_train).shape[1],
                                  np.array(x_train).shape[2])))
lstm.add(tf.keras.layers.Dropout(0.2))
lstm.add(tf.keras.layers.Dense(x_train.shape[2], activation="linear"))
# arquitectura usada
lstm.summary()

# compilar
lstm.compile(loss='mean_squared_error',
             optimizer=optimizer)
# llamar callbacks de early stopping
keras.callbacks.Callback()
stop_condition = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  mode='min',
                                                  patience=patience,
                                                  verbose=1,
                                                  min_delta=min_delta,
                                                  restore_best_weights=True)

# bajar el learning_rate durante la optimización
learning_rate_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=lr_factor,
    patience=lr_patience,
    verbose=1,
    mode="auto",
    cooldown=0,
    min_lr=lr_min)

# cuales son los callbacks que se usaran
callbacks = [stop_condition, learning_rate_schedule]
# entrenar
history = lstm.fit(x_train, y_train, validation_split=validation_size,
                   batch_size=batch_size,
                   epochs=epochs,
                   shuffle=False,
                   verbose=1,
                   callbacks=callbacks)

# mostrar el historial de entrenamiento
training_history(history, model_name="Celdas LSTM", filename="LSTM")

# predicciones
predictions = lstm.predict(x_test)
# reescalar los conjuntos a valores de unidades reales
predictions = sc.inverse_transform(predictions)
y_test = sc.inverse_transform(y_test)
# hacer predicciones siempre mayor a cero
predictions = delete_negatives(predictions)

# matriz de evaluación de la serie de tiempo
evaluation = lstm_metric_evaluation(
    predictions, y_test, fechas, names)

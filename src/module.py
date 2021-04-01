import os
import random
import warnings
# import mxnet as mx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore", category=DeprecationWarning)


def seed_everything():
    """
    Reproductibilidad de resultados
    Returns
    -------
    """
    random.seed(20)
    np.random.seed(20)
    # mx.random.seed(20)


def plot_time_series(df, fecha_inicial="2014-12-01",
                     fecha_final="2014-12-14",
                     title="Consumo Electrico",
                     ylabel="Consumo Electrico [kW]",
                     sample=10):
    """
    Plot de la serie de tiempo que tiene como indice la fehca en formato
    timestamp

    Parameters
    ----------
    df : dataframe
        datos de las series de tiempo ordenados hacia el lado.
    fecha_inicial : str
        fecha inicial en el formato %y-%m-%d. The default is "2014-12-01".
    fecha_final : TYPE
        fecha final en el formato %y-%m-%d. The default is "2014-12-14".
    title : str, optional
        titulo. The default is "Consumo Electrico".
    ylabel : TYPE, optional
        nombre de eje Y. The default is "Consumo Electrico [kW]".
    sample : TYPE, optional
        cuantas columans quieres tomar. The default is 10.
    Returns
    -------
    Gráficos en una sola plana
    """
    fig, axs = plt.subplots(5, 4, figsize=(40, 40), sharex=True)
    axx = axs.ravel()
    for i in range(0, sample):
        df[df.columns[i]].loc[fecha_inicial:fecha_final].plot(ax=axx[i])
        axx[i].set_xlabel("Fecha [dias]")
        axx[i].set_ylabel(ylabel)
        axx[i].set_title(title)
        axx[i].grid(which='minor', axis='x')


def plot_prob_forecasts(ts_entry, forecast_entry, prediction_lentgh,
                        prediction_intervals=(80.0, 95.0),
                        problem="caso_electricidad",
                        color="k"):
    """
    Plot del conjunto de testeo con intervalos de confianza definidos

    Parameters
    ----------
    ts_entry : dataframe
        dataframe con fecha y cantidad consumida.
    forecast_entry : TYPE
        DESCRIPTION.
    prediction_lentgh : TYPE
        DESCRIPTION.
    prediction_intervals : TYPE, optional
        DESCRIPTION. The default is (80.0, 95.0).

    Returns
    -------
    None.

    """
    plot_length = prediction_lentgh
    legend = ["observaciones", "mediana de la predicción"] + \
        [f"{k}% intervalo de la predicción" for k in prediction_intervals][
            ::-1]
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ts_entry[-plot_length:].plot(ax=ax, color='red')
    forecast_entry.plot(prediction_intervals=prediction_intervals,
                        color=color)
    plt.grid(which="both")
    plt.legend(legend, loc="upper left", fontsize=22)
    ax.set_ylabel("Observaciones", fontsize=25)
    ax.set_xlabel("Tiempo", fontsize=25)
    ax.set_title("Observacion vs Predicciones + IC", fontsize=30)
    plt.show()
    try_create_folder(path="images")
    problem = problem + "_" + str(ts_entry.columns[0])
    fig.savefig(f"images/{problem}.png")


def try_create_folder(path="images"):
    """
    Intentar crear carpeta
    Parameters
    ----------
    path : string
        direccion.
    """
    try:
        os.mkdir(path)
    except Exception as e:
        pass


def extract_base(string):
    """
    Extrer la base del sku
    Parameters
    ----------
    string : string
        skus.
    Returns
    -------
    base : TYPE
        DESCRIPTION.
    """
    ind = string.find("|")
    base = string[0: ind]
    return base


def drop_spaces_data(df):
    """
    sacar los espacios de columnas que podrián venir interferidas
    Parameters
    ----------
    df : dataframe
        input data
    column : string
        string sin espacios en sus columnas
    Returns
    -------
    """
    for column in df.columns:
        try:
            df[column] = df[column].str.lstrip()
            df[column] = df[column].str.rstrip()
        except Exception as e:
            print(e)
            pass
    return df


def make_empty_identifiable(value):
    """
    Parameters
    ----------
    value : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    if value == "":
        output = np.nan
    else:
        output = value
    return output


def replace_empty_nans(df):
    """
    Parameters
    ----------
    df : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    for col in df.columns:
        print("buscando vacios en:", col, "...")
        df[col] = df[col].apply(lambda x: make_empty_identifiable(x))
    return df


def df_convert_float(df):
    """
    Pasa por las columnas tratando de convertirlas a float64
    Parameters
    ----------
    df : dataframe
        df de trabajo.
    Returns
    -------
    df : dataframe
        df con las columnas númericas en float.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(float)
        except Exception as e:
            print(e)
    df.reset_index(drop=True, inplace=True)
    return df


def comentary_stationarity(p):
    """
    Comentario de la estacionaridad de las

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    comments : TYPE
        DESCRIPTION.

    """
    if p <= 0.05:
        comments =\
            "Rechaza la hipótesis nula (H0), los datos son estacionarios"
    else:
        comments =\
            "No se rechaza la hipótesis nula (H0), los datos no son estacionarios"
    return comments


def creating_rolling_dataset(data, mean_rolling=30):
    """
    Entregado un dataset, crear el mismo dataset pero a través de medias
    moviles, de tal forma de generar predicciones en ella
    Parameters
    ----------
    data : dataframe
        dataframe con las demandas.
    mean_rolling : int, optional
        tamaño de la ventana temporal para hacer las medias moviles.
        The default is 30.
    Returns
    -------
    result : dataframe
        dataframe de medias moviles de data.
    """
    result = data[["fecha"]]
    # columnas a iterar sobre
    columns = list(data.columns)
    columns.remove("fecha")
    for col in columns:
        col_name = f"rolling_{col}"
        datai = data[["fecha", col]]
        datai.set_index("fecha", inplace=True)
        datai[col_name] = datai[col].rolling(mean_rolling).mean()
        datai = datai[[col_name]]
        datai.reset_index(drop=False, inplace=True)
        result = result.merge(datai, on=["fecha"], how="outer")
    result.dropna(inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def index_date(df):
    """
    Indexar la columna de las fechas
    Parameters
    ----------
    df : dataframe
        dataframe de demandas de stock en el tiempo.
    Returns
    -------
    df : dataframe
        dataframe con la fecha como index.
    """
    df.rename(columns={"fecha": "index"}, inplace=True)
    df.set_index(["index"], inplace=True)
    return df


def stationary_adf_test(df):
    """
    Test de estacionaridad
     Augmented Dickey-Fuller puede ser una de las más utilizadas.
     Utiliza un modelo autorregresivo y optimiza un criterio de información
     a través de múltiples valores de retardo (lags) diferentes. La hipótesis
     nula de la prueba es que la serie de tiempo se puede representar
     mediante una raíz unitaria, que no es estacionaria (tiene alguna
     estructura dependiente del tiempo). La hipótesis alternativa
     (que rechaza la hipótesis nula) es que la serie de tiempo es
     estacionaria.

    * Hipótesis nula (H0): si no se rechaza, sugiere que la serie de tiempo
    tiene una raíz unitaria, lo que significa que no es estacionaria.
    Tiene alguna estructura dependiente del tiempo --> problemas.
    * Hipótesis alternativa (H1): Se rechaza la hipótesis nula; sugiere que
    la serie de tiempo no tiene una raíz unitaria, lo que significa que es
    estacionaria. No tiene una estructura dependiente del tiempo.

    Voy a interpretar este resultado utilizando el valor p de la prueba.
    Un valor p por debajo de un umbral (como 5% o 1%) sugiere que rechazamos
    la hipótesis nula (estacionaria); de lo contrario, un valor p por encima
    del umbral sugiere que no rechazamos la hipótesis nula (no estacionaria),
    este es el lo clásico en tests estadísticos.

    * Valor p> 0.05: No se rechaza la hipótesis nula (H0),
        los datos no son estacionarios
    * Valor de p <= 0.05: Rechaza la hipótesis nula (H0),
        los datos son estacionarios

    Parameters
    ----------
    df : dataframe
        dataframe al cual se le harán pruebas de estacionaridad a todas sus
        columnas.
    Returns
    -------
    output : dataframe
        resultados del test.
    """
    output = []
    for col in df.columns:
        if "fecha" in col:
            pass
        else:
            datai = df[[col]]
            # datai = datai[datai[col] > 0]
            result = adfuller(datai.values)
            # descomprimir valores
            p = result[1]
            print("Para la columna: ", col)
            print(comentary_stationarity(p))
            print('ADF estadisticas: %f' % result[0])
            print('Valor de p: %f' % result[1])
            print('Valores criticos:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))
            output.append([col, comentary_stationarity(p),
                          p, list(result[4].items())])
    output = pd.DataFrame(output, columns=["columna", "resultado",
                                           "valor_p", "intervalos"])
    return output


def downcast_dtypes(df):
    """
    Función super util para bajar la cantidad de operaciones flotante que
    se van a realizar en el proceso de entrenamiento de la red
    Parameters
    ----------
    df : dataframe
        df a disminuir la cantidad de operaciones flotantes.
    Returns
    -------
    df : dataframe
        dataframe con los tipos int16 y float32 como formato número
    """
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def lstm_preparation(array, timesteps=30):
    """
    Preparar los datos para la predicción con la lstm
    Parameters
    ----------
    array : numpy.array
        array.
    timesteps : int, optional
        cantidad de tiemsteps que se harán las predicciones.
        The default is 30.
    Returns
    -------
    x_train : array
        matriz de entrenamiento de las celdas lstm.
    y_train : array
        salida de las celdas.
    """
    x_train = []
    y_train = []
    for i in range(timesteps, array.shape[0]):
        x_train.append(array[i-timesteps:i])
        y_train.append(array[i][0:array.shape[1]])
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    return x_train, y_train


def always_number(df):
    """
    Pasa por las columnas del dataframe poniendo valores > 0
    Parameters
    ----------
    df : dataframe
        df de trabajo.
    Returns
    -------
    df : dataframe
        df con las columnas númericas en float.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(lambda x: 10e-20 if x == 0 else x)
        except Exception as e:
            print(e)
    df.reset_index(drop=True, inplace=True)
    return df


def delete_negatives(predictions):
    """
    No pueden haber predicciones negativas, asi que nos aseguramos de eso
    Parameters
    ----------
    predictions : array
        predicciones para todos los sku.
    Returns
    -------
    predictions : array
        predicciones con 0 en los casos en que estás eran negativas.
    """
    predictions = pd.DataFrame(predictions)
    for col in predictions.columns:
        predictions[col] = predictions[col].apply(
            lambda x: 0 if x < 0 else x)
    predictions = predictions.to_numpy()
    return predictions


def lstm_metric_evaluation(predictions, y_test, fechas, filter_names):
    """
    Evaluar la suma de la demanda predecida para el ciclo, con la suma de
    la demanda real en el ciclo de test
    Parameters
    ----------
    predictions : array
        predicciones.
    y_test : array
        testing.
    fechas : array
        array con las fechas de testing.
    filter names : array
        array con los nombres de los articulos.
    Returns
    -------
    mean : dataframe
        metricas de evaluación.
    """
    mean = []
    for i in range(y_test.shape[1]):
        print("columna: ", i)
        predi = predictions[:, i]
        testi = y_test[:, i]
        plot_sequence(predi, testi, fechas, filter_names[i])
        error = np.abs(predi - testi).mean()
        suma_pred = predi.sum()
        suma_test = testi.sum()
        pt = np.abs(suma_test - suma_pred) / suma_test * 100
        acc = calculate_accuracy(suma_test, suma_pred)
        mean.append([i, error, suma_pred, suma_test, pt, acc])
    mean = pd.DataFrame(mean, columns=["id_columna", "mae", "suma_pred",
                                       "suma_test", "mape", "acc"])
    # mean = delete_negatives(mean)
    # mean = pd.DataFrame(mean, columns=["id_columna", "mae", "suma_pred",
    #                                    "suma_test", "mape", "acc"])
    return mean


def calculate_accuracy(real, predict):
    """
    Calcular accuracy según metodologia de:
    https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/
    deep-learning/16.attention-is-all-you-need.ipynb
    Parameters
    ----------
    real : float, array
        real.
    predict : float, array
        predicciones.
    Returns
    -------
    TYPE
        float, array.
    """
    real = np.array(real)
    predict = np.array(predict)
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def plot_instance_training(history, epocas_hacia_atras, model_name,
                           filename):
    """
    Sacar el historial de entrenamiento de epocas en partivular
    Parameters
    ----------
    history : object
        DESCRIPTION.
    epocas_hacia_atras : int
        epocas hacia atrás que queremos ver en el entrenamiento.
    model_name : string
        nombre del modelo.
    filename : string
        nombre del archivo.
    Returns
    -------
    bool
        gráficas de lo ocurrido durante el entrenamiento.
    """
    letter_size = 20
    # Hist training
    largo = len(history.history['loss'])
    x_labels = np.arange(largo-epocas_hacia_atras, largo)
    x_labels = list(x_labels)
    # Funciones de costo
    loss_training = history.history['loss'][-epocas_hacia_atras:]
    loss_validation = history.history['val_loss'][-epocas_hacia_atras:]
    # Figura
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.plot(x_labels, loss_training, 'b', linewidth=2)
    ax.plot(x_labels, loss_validation, 'r', linewidth=2)
    ax.set_xlabel('Epocas', fontname="Arial", fontsize=letter_size-5)
    ax.set_ylabel('Función de costos', fontname="Arial",
                  fontsize=letter_size-5)
    ax.set_title(f"{model_name}", fontname="Arial", fontsize=letter_size)
    ax.legend(['Entrenamiento', 'Validación'], loc='upper left',
              prop={'size': letter_size-5})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size-5)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size-5)
    plt.show()
    return fig


def training_history(history, model_name="Celdas LSTM", filename="LSTM"):
    """
    Según el historial de entrenamiento que hubo plotear el historial
    hacía atrás de las variables
    Parameters
    ----------
    history : list
        lista con errores de validación y training.
    model_name : string, optional
        nombre del modelo. The default is "Celdas LSTM".
    filename : string, optional
        nombre del archivo. The default is "LSTM".

    Returns
    -------
    None.

    """
    size_training = len(history.history['val_loss'])
    fig = plot_instance_training(history, size_training, model_name,
                                 filename + "_ultimas:" +
                                 str(size_training) + "epocas")

    fig = plot_instance_training(history, int(1.5 * size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" +
                                 str(1.5 * size_training / 2) + "epocas")
    # guardar el resultado de entrenamiento de la lstm
    fig.savefig(f"results/{model_name}_training.png")

    fig = plot_instance_training(history, int(size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 2) + "epocas")

    fig = plot_instance_training(history, int(size_training / 3), model_name,
                                 filename + "_ultimas:" +
                                 str(size_training / 3) + "epocas")
    fig = plot_instance_training(history, int(size_training / 4), model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 4) + "epocas")
    print(fig)


def plot_sequence(predictions, real, fechas, indice):
    """
    Plot sequence de la secuecnia
    Parameters
    ----------
    predictions : array
        predicciones.
    real : array
        valores reales.
    fechas : array
        array de fechas.
    indice : TYPE
        indice de la columna.
    Returns
    -------
    plot de prediciones vs real.
    """
    letter_size = 20
    new_fechas = []
    for fecha in fechas:
        fecha = fecha[0:10]
        new_fechas.append(fecha)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, figsize=(20, 12))
    ax.plot(new_fechas, real, 'gold', linewidth=2)
    ax.plot(new_fechas, predictions, 'orangered', linewidth=2)
    ax.set_xlabel('Tiempo', fontname="Arial", fontsize=letter_size)
    ax.set_ylabel('Predicción vs Real', fontname="Arial",
                  fontsize=letter_size+2)
    ax.set_title(f"Predicciones vs real {str(indice)}",
                 fontname="Arial", fontsize=letter_size+10)
    ax.legend(['real', 'predicción'], loc='upper left',
              prop={'size': letter_size+5})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size)
    try_create_folder("results/lstm")
    plt.xticks(rotation=75)
    plt.show()
    fig.savefig(f"results/lstm/{indice}_results.png")


def grouping_df(df, days=10, arg="sum"):
    """
    Dado un dataframe que posee una columna de

    Parameters
    ----------
    df : dataframe
        dataframe a agrupar en una fecha dada.
    days : int, optional
        dias de la fecha. The default is 10.
    arg : string, optional
        sum o mean como argumentos. The default is "sum".
    Returns
    -------
    df : dataframe
        dataframe agrupado en esos dias.
    """
    df["fecha"] = df["fecha"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H-%M-%S"))
    df["fecha_aprox"] = df["fecha"].apply(lambda x: x.round(f"{days}d"))
    if arg == "sum":
        print("Se agrupa por suma")
        df = df.groupby("fecha_aprox").sum()
    elif arg == "mean":
        print("Se agrupa por promedio")
        df = df.groupby("fecha_aprox").mean()
    else:
        print("Argumentos no validos, no se ha efectuado la operación")
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"fecha_aprox": "fecha"}, inplace=True)
    df["fecha"] = df["fecha"].apply(
        lambda x: x.strftime("%Y-%m-%d %H-%M-%S"))
    df.reset_index(drop=True, inplace=True)
    return df


def plot_prob_forecasts_deep_renewal(ts_entry, forecast_entry, title):
    """
    Plot del conjunto de testeo con intervalos de confianza definidos

    Parameters
    ----------
    ts_entry : dataframe
        dataframe con fecha y cantidad consumida.
    forecast_entry : TYPE
        DESCRIPTION.
    prediction_lentgh : TYPE
        DESCRIPTION.
    prediction_intervals : TYPE, optional
        DESCRIPTION. The default is (80.0, 95.0).

    Returns
    -------
    None.

    """
    plot_length = 100
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + \
        [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    title =\
        title if forecast_entry.item_id is None else title +\
        "|"+forecast_entry.item_id
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.title(title)
    plt.show()

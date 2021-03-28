import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

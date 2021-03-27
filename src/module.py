import os
import numpy as np
import matplotlib.pyplot as plt


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
    fig.savefig(f"images/{problem}.png", dpi=200)


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

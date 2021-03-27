import os
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
    fig, axs = plt.subplots(10, 4, figsize=(40, 40), sharex=True)
    axx = axs.ravel()
    for i in range(0, sample):
        df[df.columns[i]].loc[fecha_inicial:fecha_final].plot(ax=axx[i])
        axx[i].set_xlabel("Fecha [dias]")
        axx[i].set_ylabel(ylabel)
        axx[i].set_title(title)
        axx[i].grid(which='minor', axis='x')


def plot_prob_forecasts(ts_entry, forecast_entry, prediction_lentgh,
                        prediction_intervals=(80.0, 95.0),
                        problem="caso_electricidad"):
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
                        color='k')
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

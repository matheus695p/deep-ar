import warnings
import pandas as pd
from datetime import datetime
from src.module import (plot_time_series, creating_rolling_dataset,
                        index_date, stationary_adf_test)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("data/manufacturing.csv")
fecha_inicial = df["fecha"].iloc[0][0: 10]
fecha_final = df["fecha"].iloc[-1][0: 10]
print("Fecha inicial del historial de ventas: ", fecha_inicial)
print("Fecha final del historial de ventas: ", fecha_final)
# dejar en formato adecuado
df["fecha"] = df["fecha"].apply(lambda x:
                                datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
# dejando una copia para el dataset de medias moviles
data = df.copy()
# indexar la fecha en el dataset
df = index_date(df)
# plot de la serie temporal
plot_time_series(df, fecha_inicial="2019-03-12", fecha_final="2019-05-12",
                 title="Unidades Vendidas",
                 ylabel="Unidades vendidas [u]",
                 sample=19)
# resultados de ADF test
output = stationary_adf_test(df)

# dataset de medias moviles
df_30 = creating_rolling_dataset(data, mean_rolling=30)
# guardar la data con una ventana temporal de 30
df_30.to_csv("data/manufacturing_r30.csv", index=False,
             date_format='%Y-%m-%d %H:%M:%S')
# indexar dataframe
df_30 = index_date(df_30)
# plot de las medias moviles
plot_time_series(df_30,
                 fecha_inicial="2019-04-12", fecha_final="2019-12-12",
                 title="Unidades Vendidas por medias moviles",
                 ylabel="Unidades vendidas [u]",
                 sample=19)
# verificar estacionaridad con ADF
output_30 = stationary_adf_test(df_30)

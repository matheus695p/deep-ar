import warnings
import pandas as pd
from datetime import datetime
from src.module import (df_convert_float, extract_base, drop_spaces_data,
                        replace_empty_nans)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# lectura de los datos en .txt
df = pd.read_csv('data/raw_manufacturing.csv', index_col=0)
# limpieza del dataset
df = drop_spaces_data(df)
df = replace_empty_nans(df)
df = df_convert_float(df)

# sacar el componente base del sku
df["cantidad_vendida"] = df["cantidad_vendida"].apply(
    lambda x: 0 if x < 0 else x)
df = df[df["cantidad_vendida"] > 0]
df["base_sku"] = df["sku"].apply(lambda x: extract_base(x))
df["fecha"] = df["fecha"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
df = df[["base_sku", "fecha", "cantidad_vendida"]]

# acumular en un solo dia las ventas
df = df.groupby(["fecha", "base_sku"]).sum()
df.reset_index(drop=False, inplace=True)

# selecionar los sku con más registros por sobre lim de ventas
lim = 200
skus = pd.DataFrame(df["base_sku"].value_counts())
skus.reset_index(drop=False, inplace=True)
skus.columns = ["base_sku", "cantidad_registros"]
skus = skus[skus["cantidad_registros"] >= lim]
bases = skus["base_sku"].to_list()
print("Bases de SKU's escogidos para el trabajo:", len(bases))

# columnas del dataframe
result = df[["fecha"]]
result.drop_duplicates(inplace=True)
result.reset_index(drop=True, inplace=True)

# crear el dataset
for base in bases:
    print(base)
    datai = df[df["base_sku"] == base]
    datai.reset_index(drop=True, inplace=True)
    datai = datai[["fecha", "cantidad_vendida"]]
    datai.rename(columns={"cantidad_vendida": base}, inplace=True)
    result = result.merge(datai, on="fecha", how="outer")

# resampleo a 1D
result.set_index(["fecha"], inplace=True)
result = result.resample("1D").sum()
result.reset_index(drop=False, inplace=True)
result.fillna(value=0, inplace=True)
result.sort_values(by=["fecha"], inplace=True)
result.reset_index(drop=True, inplace=True)

# guardar la data de manufactura
result.to_csv("data/manufacturing.csv", index=False,
              date_format='%Y-%m-%d %H:%M:%S')

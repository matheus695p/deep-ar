import warnings
import pandas as pd
import matplotlib as mpl
from pathlib import Path
from tqdm.autonotebook import tqdm
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from src.module import (plot_time_series, plot_prob_forecasts,
                        try_create_folder)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# seteo de matplotlib
mpl.rcParams['figure.figsize'] = (20, 12)
mpl.rcParams['axes.grid'] = False

# lectura de los datos en .txt
df = pd.read_csv('data/LD2011_2014.txt', sep=';', index_col=0,
                 parse_dates=True, decimal=',')
# copia del dataframe
data = df.copy()
data.reset_index(drop=False, inplace=True)
# explorar el dataset
head = df.head(5000)
# ver la serie de tiempo completa
plot_time_series(df, fecha_inicial="2014-12-01", fecha_final="2014-12-14",
                 title="Consumo Electrico",
                 ylabel="Consumo Electrico [kW]",
                 sample=20)

# poner en el formato que requiere gluonts
df_input = df.reset_index(drop=True).T.reset_index()
# indice de los lugares de consumo electrico
ts_code = df_input["index"].astype('category').cat.codes.values

# separar los datos
test_index = int(len(df) * 0.8)
# fecha de inicio
start_train = str(data["index"].iloc[0])[0:19]
start_test = str(data["index"].iloc[test_index])[0:19]

# test_index = 134999
df_train = df_input.iloc[:, 1:test_index].values
df_test = df_input.iloc[:, test_index:].values
print("foma de conjunto entrenamiento", df_train.shape)
print("foma de conjunto test", df_test.shape)

# frecuencia de los timestamps para pandas en minutos
freq = "15min"
# frecuencia en horas
freq_int = 60 / int(''.join(filter(str.isdigit, freq)))
# a una freuencia de 15 minutos
days_of_prediction = 7
# largo de la prediccion para que sean days_of_prediction dias
prediction_lentgh = int(days_of_prediction * 24 * freq_int)
# n??mero de columnas con el que se har??n las predicciones (simil: productos)
number_of_products = 15

# fechas de inicio de los conjuntos
start_train = pd.Timestamp(start_train, freq=freq)
start_test = pd.Timestamp(start_test, freq=freq)

# settings del modelo, ver link para conocer los hyperparametros
# https://ts.gluon.ai/api/gluonts/gluonts.model.deepar.html
estimator = DeepAREstimator(freq=freq,
                            context_length=prediction_lentgh,
                            prediction_length=prediction_lentgh,
                            use_feat_static_cat=True,
                            cardinality=[1],
                            num_layers=2,
                            num_cells=128,
                            cell_type='lstm',
                            trainer=Trainer(epochs=15))

# reshape de la data de entrenamiento para solo ocupar number_of_products
train_ds = ListDataset([{
    FieldName.TARGET: target,
    FieldName.START: start_train,
    FieldName.FEAT_STATIC_CAT: fsc}
    for (target, fsc) in zip(df_train[0:number_of_products],
                             ts_code[0:number_of_products].reshape(-1, 1))],
    freq=freq)

# reshape de la data de test para solo ocupar number_of_products
test_ds = ListDataset([{
    FieldName.TARGET: target,
    FieldName.START: start_test,
    FieldName.FEAT_STATIC_CAT: fsc}
    for (target, fsc) in zip(df_test[0:number_of_products],
                             ts_code[0:number_of_products].reshape(-1, 1))],
    freq=freq)

# entrenar el predictor
predictor = estimator.train(training_data=train_ds)

# evaluar las predicciones para el conjunto de testing
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100)

print("Obtenci??n de valores de acondicionamiento de series de tiempo ...")
tss = list(tqdm(ts_it, total=len(df_test)))
print("Obtenci??n de valores de acondicionamiento de series de tiempo ...")
forecasts = list(tqdm(forecast_it, total=len(df_test)))

# plotear las predicciones en un intervalo de confianza
for i in tqdm(range(number_of_products - 1)):
    ts_entry = tss[i]
    ts_entry.columns = [list(df.columns)[i]]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry, prediction_lentgh)

# evaluar para los cuantiles
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(
    iter(tss), iter(forecasts),
    num_series=len(df_test[0:number_of_products]))

# formateo del nombre
columns = pd.DataFrame(list(df.columns)[0: number_of_products],
                       columns=["name"])
item_metrics["item_id"] = columns["name"]

item_metrics.drop(columns=["OWA"], inplace=True)
print(item_metrics)

# guardar par??metros
path = "results/"
try_create_folder(path)
path = path + "energy-model"
try_create_folder(path)

# guardar los resultados
item_metrics.to_csv("results/resultados_energia.csv", index=False)

# guardar el modelo
predictor.serialize(Path(path))

# hacer inferencia con el modelo guardado
predictor_deserialized = Predictor.deserialize(Path(path))

# repetir los pasos para hacer la inferencia
print("...")

# import os
# import ast
# import random
# import joblib
import warnings
import mxnet as mx
import numpy as np
import pandas as pd
import matplotlib as mpl
# import plotly.express as px
from tqdm import tqdm
# from pathlib import Path
from datetime import datetime, timedelta
# from deeprenewal import get_dataset
from gluonts.trainer import Trainer
# from gluonts.evaluation import Evaluator
# from gluonts.dataset.util import to_pandas
from deeprenewal import DeepRenewalEstimator
# from gluonts.model.npts import NPTSPredictor
# from gluonts.model.predictor import Predictor
# from deeprenewal import IntermittentEvaluator
from gluonts.dataset.common import ListDataset
# from deeprenewal import CrostonForecastPredictor
# from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.field_names import FieldName
# from gluonts.model.forecast import SampleForecast
# from gluonts.model.prophet import ProphetPredictor
# from gluonts.model.r_forecast import RForecastPredictor
# from gluonts.distribution.student_t import StudentTOutput
from gluonts.evaluation.backtest import make_evaluation_predictions
# from gluonts.distribution.neg_binomial import NegativeBinomialOutput
# from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from src.module import (index_date, seed_everything)
from src.config import arguments_parser
warnings.filterwarnings("ignore", category=DeprecationWarning)
# semillas
seed_everything()

# argumentos de entrenamiento
args = arguments_parser()
is_gpu = mx.context.num_gpus() > 0

# seteo de matplotlib
mpl.rcParams['figure.figsize'] = (20, 12)
mpl.rcParams['axes.grid'] = False

# lectura de los datos en .txt
df = pd.read_csv('data/manufacturing.csv', parse_dates=True)

# fechas inicial y final
fecha_inicial = df["fecha"].iloc[0][0: 10]
fecha_final = df["fecha"].iloc[-1][0: 10]

# buscar indice de finalizacion
end_date = datetime.strptime(df["fecha"].iloc[-1],
                             "%Y-%m-%d %H-%M-%S") - timedelta(days=14)
end_date = end_date.strftime("%Y-%m-%d %H-%M-%S")
# separar los datos
test_index = df[df["fecha"] == end_date].index[0]

# a una freuencia de 15 minutos
days_of_prediction = len(df) - test_index
# start_date
start_date = df["fecha"].iloc[0][0: 10]
end_date = df["fecha"].iloc[test_index][0: 10]

print("Fecha inicial del historial de ventas: ", fecha_inicial)
print("Fecha final del historial de ventas: ", fecha_final)
# convertir fecha a timestamp
df["fecha"] = df["fecha"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H-%M-%S"))
# indexar fecha
df = index_date(df)

# poner en el formato que requiere gluonts
df_input = df.reset_index(drop=True).T.reset_index()

# indice de los lugares de consumo electrico
ts_code = df_input["index"].astype('category').cat.codes.values

# test_index = 134999
df_train = df_input.iloc[:, 1:test_index].values
df_test = df_input.iloc[:, test_index:].values
print("foma de conjunto entrenamiento", df_train.shape)
print("foma de conjunto test", df_test.shape)

# frecuencia de los timestamps para pandas en minutos
freq = "1440min"
# frecuencia en horas
freq_int = 60 / int(''.join(filter(str.isdigit, freq)))

# largo de la prediccion para que sean days_of_prediction dias
prediction_length = int(days_of_prediction * 24 * freq_int)
# número de columnas con el que se harán las predicciones (simil: productos)
number_of_products = 19

# fechas de inicio de los conjuntos
start_train = pd.Timestamp(start_date + " 00:00:00", freq=freq)
start_test = pd.Timestamp(end_date + " 00:00:00", freq=freq)


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
    FieldName.START: end_date,
    FieldName.FEAT_STATIC_CAT: fsc}
    for (target, fsc) in zip(df_test[0:number_of_products],
                             ts_code[0:number_of_products].reshape(-1, 1))],
    freq=freq)

next(iter(train_ds))


# estimador de deep renewal
trainer = Trainer(
    ctx=mx.context.gpu() if is_gpu & args.use_cuda else mx.context.cpu(),
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    epochs=20,
    num_batches_per_epoch=args.number_of_batches_per_epoch,
    clip_gradient=args.clip_gradient,
    weight_decay=args.weight_decay,
    hybridize=False)

estimator = DeepRenewalEstimator(
    prediction_length=prediction_length,
    context_length=prediction_length*args.context_length_multiplier,
    num_layers=args.num_layers,
    num_cells=args.num_cells,
    cell_type=args.cell_type,
    dropout_rate=args.dropout_rate,
    scaling=True,
    lags_seq=np.arange(1, args.num_lags+1).tolist(),
    freq=freq,
    use_feat_dynamic_real=args.use_feat_dynamic_real,
    use_feat_static_cat=args.use_feat_static_cat,
    use_feat_static_real=args.use_feat_static_real,
    trainer=trainer)
predictor = estimator.train(train_ds)

# Evaluación de los 3 casos

# Deep Renewal Flat
deep_renewal_flat_forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds, predictor=predictor, num_samples=100)
deep_renewal_flat_forecasts = list(
    tqdm(deep_renewal_flat_forecast_it, total=len(test_ds)))

# Deep Renewal Exact
predictor.forecast_generator.forecast_type = "exact"
deep_renewal_exact_forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds, predictor=predictor, num_samples=100)
deep_renewal_exact_forecasts = list(
    tqdm(deep_renewal_exact_forecast_it, total=len(test_ds)))

# #Deep Renewal Hybrid
predictor.forecast_generator.forecast_type = "hybrid"
deep_renewal_hybrid_forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds, predictor=predictor, num_samples=100)
deep_renewal_hybrid_forecasts = list(
    tqdm(deep_renewal_hybrid_forecast_it, total=len(test_ds)))

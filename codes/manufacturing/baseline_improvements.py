import os
import ast
import random
import joblib
import mxnet as mx
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from gluonts.evaluation.backtest import make_evaluation_predictions
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from argparse import ArgumentParser
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.model.npts import NPTSPredictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.distribution.student_t import StudentTOutput
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.model.deepar import DeepAREstimator
from deeprenewal import IntermittentEvaluator
from deeprenewal import CrostonForecastPredictor
from deeprenewal import DeepRenewalEstimator
from gluonts.dataset.util import to_pandas
from deeprenewal import get_dataset


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    mx.random.seed(42)


seed_everything()


parser = ArgumentParser()
parser.add_argument(
    "-f", "--fff", help="a dummy argument to fool ipython", default="1")

# add donde correr y guardar datos
parser.add_argument('--use-cuda', type=bool, default=True)
parser.add_argument('--log-gradients', type=bool, default=True)
parser.add_argument('--datasource', type=str, default="retail_dataset")
parser.add_argument('--model-save-dir', type=str, default="saved_models")

# trainer
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=1e-2)
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--number-of-batches-per-epoch', type=int, default=100)
parser.add_argument('--clip-gradient', type=float, default=5.170127652392614)
parser.add_argument('--weight-decay', type=float, default=0.01)

# args modelo
parser.add_argument('--context-length-multiplier', type=int, default=2)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--num-cells', type=int, default=64)
parser.add_argument('--cell-type', type=str, default="lstm")

# p% are dropped and set to zero
parser.add_argument('--dropout-rate', type=float, default=0.3)
parser.add_argument('--use-feat-dynamic-real', type=bool, default=False)
parser.add_argument('--use-feat-static-cat', type=bool, default=False)
parser.add_argument('--use-feat-static-real', type=bool, default=False)
parser.add_argument('--scaling', type=bool, default=True)
parser.add_argument('--num-parallel-samples', type=int, default=100)
parser.add_argument('--num-lags', type=int, default=1)

# Only for Deep Renewal Processes
parser.add_argument('--forecast-type', type=str, default="hybrid")

# Only for Deep AR
parser.add_argument('--distr-output', type=str,
                    default="student_t")  # neg_binomial
args = parser.parse_args()
is_gpu = mx.context.num_gpus() > 0


# lectura del dataset
dataset = get_dataset(args.datasource, regenerate=False)

# metadata del dataset
prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
cardinality = ast.literal_eval(
    dataset.metadata.feat_static_cat[0].cardinality)
# sacar conjuntos de entrenamiento
train_ds = dataset.train
test_ds = dataset.test

# predictor ets
ets_predictor = RForecastPredictor(freq=freq,
                                   prediction_length=prediction_length,
                                   method_name='ets')
ets_forecast = list(ets_predictor.predict(train_ds))


# arima predictor
arima_predictor = RForecastPredictor(freq=freq,
                                     prediction_length=prediction_length,
                                     method_name='arima')
arima_forecast = list(arima_predictor.predict(train_ds))

# predictor croston
croston_predictor = CrostonForecastPredictor(
    freq=freq, prediction_length=prediction_length,
    variant='original',
    no_of_params=2)
croston_forecast = list(croston_predictor.predict(train_ds))

# sba predictor
sba_predictor = CrostonForecastPredictor(freq=freq,
                                         prediction_length=prediction_length,
                                         variant='sba',
                                         no_of_params=2)
sba_forecast = list(sba_predictor.predict(train_ds))

# sbj
sbj_predictor = CrostonForecastPredictor(freq=freq,
                                         prediction_length=prediction_length,
                                         variant='sbj',
                                         no_of_params=2)
sbj_forecast = list(sbj_predictor.predict(train_ds))

# npts predictor
npts_predictor = NPTSPredictor(freq=freq,
                               prediction_length=prediction_length,
                               context_length=300, kernel_type='uniform',
                               use_seasonal_model=False)
npts_forecast = list(npts_predictor.predict(train_ds))


# deep ar
distr = PiecewiseLinearOutput(7)
deep_ar_trainer = Trainer(
    ctx=mx.context.gpu() if is_gpu & args.use_cuda else mx.context.cpu(),
    batch_size=128,
    learning_rate=1e-2,
    epochs=20,
    num_batches_per_epoch=args.number_of_batches_per_epoch,
    clip_gradient=5.48481845049343,
    weight_decay=0.001,
    hybridize=False)

deep_ar_estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length*2,
    num_layers=2,
    num_cells=128,
    cell_type='gru',
    dropout_rate=0.1,
    scaling=True,
    lags_seq=np.arange(1, 1+1).tolist(),
    freq=freq,
    use_feat_dynamic_real=False,
    use_feat_static_cat=False,
    use_feat_static_real=False,
    distr_output=distr,
    cardinality=None,
    trainer=deep_ar_trainer)
deep_ar_predictor = deep_ar_estimator.train(train_ds, test_ds)


print("Generating Deep AR forecasts.......")
deep_ar_forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds, predictor=deep_ar_predictor, num_samples=100)
tss = list(tqdm(ts_it, total=len(test_ds)))
deep_ar_forecasts = list(tqdm(deep_ar_forecast_it, total=len(test_ds)))


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
    cardinality=cardinality if args.use_feat_static_cat else None,
    trainer=trainer)
predictor = estimator.train(train_ds, test_ds)


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

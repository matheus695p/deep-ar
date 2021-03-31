import warnings
import pandas as pd
import matplotlib as mpl
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.autonotebook import tqdm
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from src.module import (plot_time_series, plot_prob_forecasts,
                        try_create_folder, index_date)
warnings.filterwarnings("ignore", category=DeprecationWarning)
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

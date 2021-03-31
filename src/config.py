import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)


def arguments_parser():
    # argumentos
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="a dummy argument to fool ipython", default="1")
    # agregar donde correr y guardar datos
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--log-gradients', type=bool, default=True)
    parser.add_argument('--datasource', type=str, default="retail_dataset")
    parser.add_argument('--model-save-dir', type=str, default="saved_models")
    # objecto trainer
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--number-of-batches-per-epoch',
                        type=int, default=100)
    parser.add_argument('--clip-gradient', type=float,
                        default=5.170127652392614)
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
    # solo para Deep Renewal Processes
    parser.add_argument('--forecast-type', type=str, default="hybrid")
    # solo para Deep AR
    # neg_binomial
    parser.add_argument('--distr-output', type=str,
                        default="student_t")
    args = parser.parse_args()
    return args

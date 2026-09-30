import argparse
from fno.models import *

# Available model types
MODEL_TYPES = ["fno1d"]

# Uses argparse to read arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model",
    help=f"Model type that you wish to create. Options are {MODEL_TYPES}",
    type=str,
    required=True
)

parser.add_argument(
    "-p", "--path",
    help=f"Path where the model should be saved and filename",
    type=str,
    required=True
)

# Based on the chosen model type, adds the other required arguments to create it
args, _ = parser.parse_known_args()

if args.model == "fno1d":
    fno1d.set_create_arguments(parser=parser)

# Parses the rest of the required arguments
args = parser.parse_args()

if args.model == "fno1d":
    model = fno1d.from_args(args)
    model.saveModel(args.path)
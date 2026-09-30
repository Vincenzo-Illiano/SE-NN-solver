import argparse
from fno.models import *
from fno import training
from fno import losses
from torch import optim
import torch
import fno

# Uses argparse to read arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model",
    help=f"Model type that you wish to train",
    type=str,
    required=True
)

parser.add_argument(
    "-p", "--modelpath",
    help=f"Path of the model's checkpoint file",
    type=str,
    required=True
)

parser.add_argument(
    "-t", "--trainset",
    help=f"Path of the training set",
    type=str,
    required=True
)

parser.add_argument(
    "-s", "--save",
    help=f"Path where you want to save the trained model. If left empty, the existing model will be overwritten",
    type=str,
)

parser.add_argument(
    "-e", "--epochs",
    help="Number of epochs that the model will use for training",
    type=int,
    required=True
)

parser.add_argument(
    "-l", "--loss",
    help=f"Loss function that will be used during the training. Available options are {fno.LOSSES}",
    type=str,
    required=True
)

parser.add_argument(
    "-d", "--device",
    help=f"Device where the training is done. Could be cpu or cuda. Default is cuda",
    type=str,
    default="cuda"
)

parser.add_argument(
    "--noProgress",
    help="Use this if you don't want training progress to be print in the console",
    action="store_true"
)

parser.add_argument(
    "--shuffleset",
    help="Whether to shuffle the dataset after each epoch",
    action="store_true"
)

# Based on the chosen model type, adds the other required arguments to create it
args = parser.parse_args()

# If no save directory is specified, the existing one will be overwritten
if args.save is None:
    save = args.modelpath

# Loads the training set
trainset = torch.load(args.trainset)

# Loads the loss function
if args.loss == "minimumE_loss":
    loss = losses.minimumE_loss
if args.loss == "minimumErr_loss":
    loss = losses.minimumErr_loss

# For each model type, trains the model
if args.model == "fno1d":

    # Loads the model
    model = fno1d.from_file(args.modelpath)

    # At the moment, only one optimizer will be used and there are not many options
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Trains the model
    model, _ = training.trainFno1d(model, trainset, optimizer, args.epochs, loss, args.device, not args.noProgress, args.shuffleset)

    # Saves the trained model
    model.saveModel(save)
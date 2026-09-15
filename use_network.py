import torch
from network import network
from network import testing
from network import training
from pathlib import Path
import argparse

# Finds the root folder and checkpoint path 
ROOT = Path(__file__).resolve().parent
statedict_path = ROOT / "network" / "data" / "checkpoint.pth"
testing_path = ROOT / "network" / "data" / "testingset.pt"
training_path = ROOT / "network" / "data" / "trainingset.pt"
testinfo_path = ROOT / "testinfo.txt"

# Uses argparse to read arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model",
    help="Path of the model state dict. Default will be /data/checkpoint.pth",
    default=statedict_path,
    type=str
)


parser.add_argument(
    "-t", "--train",
    nargs="?",
    const=training_path,
    help="Path of the training set. Default will be /data/trainingset.pt. Don't use this flag if you don't want to train the model",
    default=None,
    type=str,
)

parser.add_argument(
    "-T", "--test",
    nargs="?",
    const=testing_path,
    help="Path of the testing set. Default will be /data/testingset.pt. Don't use this flag if you don't want to test the model",
    default=None,
    type=str,
)

parser.add_argument(
    "-p", "--plot",
    type=int,
    default=1,
    help="How many testing result plots to generate. Only works when using --test."
)


parser.add_argument(
    "-g", "--gpu",
    action="store_true",
    help="Use dedicated gpu if available"
)

args = parser.parse_args()

# Checks if the model will be trained / tested based on the arguments
istrain = args.train is not None
training_path = args.train if istrain else None
istest = args.test is not None
testing_path = args.test if istest else None

# Updates paths based on arguments
statedict_path = args.model

if(args.plot < 1):
    parser.error("Number of plot cannot be less than 1")


# checks if the given model exists
if(Path(statedict_path).exists() == False):
    if(input(f"File {statedict_path} doesn't exist. Do you want to create a new model? (y/N)") =='N'):
        parser.error(f"File {statedict_path} doesn't exist. Can't load network")
    print(f"A new model will be generated in {statedict_path} with random parameters")

    # Creates the model
    model = network.FourierNet()

    # Saves the state dict
    torch.save(model.state_dict(), statedict_path)

    print(f"Created model successfully in file {statedict_path}")


if(istrain == False and istest == False):
    print("There is nothing to do")
    exit()

# Checks if the given testing path exists
if(istest and Path(testing_path).exists() == False):
    parser.error(f"File {testing_path} doesn't exist. Can't train network")
    
if(istrain and Path(training_path).exists() == False):
    parser.error(f"File {training_path} doesn't exist. Can't train network")

# Uses the GPU if available
device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
print("Using device: " + str(device))

# Network training
if(istrain):
    showInfo = False
    if(input("Do you want training loss to be plotted? (y/N):").lower() == 'y'):
        showInfo = True

    print("Training network...")
    training.trainNetwork(device, statedict_path, training_path, showInfo)
    print("Training done successfully.")

if(istest):
    print("Testing network...")
    testing.testNetwork(device, statedict_path, testing_path, args.plot)
    
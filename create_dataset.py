from network import dataset
import torch
from pathlib import Path
import argparse

# Finds the root folder and checkpoint path 
ROOT = Path(__file__).resolve().parent

# Finds default paths
testing_path = ROOT / "network" / "data" / "testingset.pt"
training_path = ROOT / "network" / "data" / "trainingset.pt"

# Uses argparse to read arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-T", "--testset",
    nargs="?",
    const=testing_path,
    help="Path of the testing set to generate. Default will be /data/testingset.pt.",
    default=None,
    type=str,
)

parser.add_argument(
    "-t", "--trainset",
    nargs="?",
    const=training_path,
    help="Path of the training set to generate. Default will be /data/trainingset.pt.",
    default=None,
    type=str,
)

parser.add_argument(
    "-o", "--overwrite",
    action="store_true",
    help="Overwrites existing files"
)


parser.add_argument(
    "-g", "--gpu",
    action="store_true",
    help="Use dedicated gpu if available"
)

parser.add_argument(
    "-s", "--settype",
    type=str,
    default="mixed",
    help="Type of dataset to create. Can be \"mixed\", \"poly\", \"smooth\" and \"well\""
)

args = parser.parse_args()

# Checks if the train / test sets will be generated 
istrain = args.trainset is not None
training_path = args.trainset if istrain else training_path
istest = args.testset is not None
testing_path = args.testset if istest else testing_path

# Checks if the files already exist
if(args.overwrite == False and Path(testing_path).exists() == True and istest):
    parser.error(f"file {testing_path} already exists. Use the -o flag to overwrite it")
if(args.overwrite == False and Path(training_path).exists() == True and istrain):
    parser.error(f"file {training_path} already exists. Use the -o flag to overwrite it")

if(istrain == False and istest == False):
    print("There is nothing to do")
    exit()

# Uses the GPU if available
device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
print("Using device: " + str(device))

# Generates the training set
if(istrain):
    dataset.generate_training_set(training_path, device, args.settype)
 
# Generates the testing set
if(istest):
    dataset.generate_testing_set(testing_path, device, args.settype)
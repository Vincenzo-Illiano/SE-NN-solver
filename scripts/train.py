import argparse
from pathlib import Path

import torch
from torch import optim

import fno
from fno import losses, training
from fno.models.fno1d import fno1d

def main():
    """Train an initialized model on a saved dataset of potentials."""

    # All the required arguments
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("-m", "--model", choices=["fno1d"], required=True)
    parser.add_argument(
        "-p", "--modelpath", type=Path, required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "-t", "--trainset", type=Path, required=True,
        help="Path to the training dataset.",
    )
    parser.add_argument(
        "-s", "--save", type=Path,
        help="Path for the trained model (default: overwrite the input checkpoint).",
    )
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-l", "--loss", choices=fno.LOSSES, required=True)
    parser.add_argument(
        "-d", "--device", default="cuda",
        help="Training device, for example cpu, cuda, or cuda:1 (default: cuda).",
    )
    parser.add_argument(
        "--legacy-width", type=float,
        help="Interval half-width A for a checkpoint saved as a bare state dict.",
    )
    parser.add_argument(
        "--noProgress", action="store_true",
        help="Suppress the progress counter.",
    )
    parser.add_argument(
        "--shuffleset", action="store_true",
        help="Shuffle the potentials across batches before each epoch.",
    )
    args = parser.parse_args()

    # Checks
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")

    save = args.save if args.save is not None else args.modelpath

    try:
        # Move the model before giving its parameters to the optimizer.
        model = fno1d.from_file(args.modelpath, legacy_A=args.legacy_width).to(args.device)
        trainset = torch.load(args.trainset, map_location="cpu", weights_only=True)

        # Trains
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        model, _ = training.trainFno1d(
            model, trainset, optimizer, args.epochs, getattr(losses, args.loss),
            device=args.device, printprogress=not args.noProgress,
            shuffledataset=args.shuffleset,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    # Saves the model
    save.parent.mkdir(parents=True, exist_ok=True)
    model.saveModel(save)

if __name__ == "__main__":
    main()
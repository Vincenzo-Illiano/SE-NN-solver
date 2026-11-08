import argparse
import math
from pathlib import Path

import torch

from fno import testing
from fno.models.fno1d import fno1d

def main():
    """Evaluate a saved model against a labeled testing set."""

    parser = argparse.ArgumentParser(description=main.__doc__)

    # All the arguments
    parser.add_argument("-m", "--model", choices=["fno1d"], default="fno1d")
    parser.add_argument(
        "-p", "--modelpath", type=Path, required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "-T", "--testset", type=Path, required=True,
        help="Path to the labeled testing dataset.",
    )
    parser.add_argument(
        "-d", "--device", choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device (default: CUDA when available, otherwise CPU).",
    )
    parser.add_argument(
        "-a", "--energy-weight", type=float, default=1.0,
        help="Weight of the energy MSE in the combined loss (default: 1).",
    )
    parser.add_argument(
        "--legacy-width", type=float,
        help="Interval half-width A for a checkpoint saved as a bare state dict.",
    )
    parser.add_argument(
        "--noProgress", action="store_true",
        help="Suppress the progress counter.",
    )
    args = parser.parse_args()

    # Checks if cuda is available
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")

    # Checks that the weights are finite and non-negative
    if not math.isfinite(args.energy_weight) or args.energy_weight < 0:
        parser.error("--energy-weight must be finite and non-negative.")

    # Checks that the model exists
    for path in (args.modelpath, args.testset):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")

    # Keep the dataset on the CPU and move only the current batch to the device.
    try:
        model = fno1d.from_file(args.modelpath, legacy_A=args.legacy_width)
        testset = torch.load(args.testset, map_location="cpu", weights_only=True)
        print(f"Using device: {args.device}")
        losses = testing.testFno1d(
            model, testset, a=args.energy_weight,
            device=args.device, printprogress=not args.noProgress,
        )
    except ValueError as error:
        parser.error(str(error))

    # All batches in a generated testing set contain the same number of samples.
    mean_loss = torch.stack(losses).mean().item()
    if not math.isfinite(mean_loss):
        parser.error("Test loss is non-finite. Check the checkpoint and testing data.")
    print(f"Mean test loss: {mean_loss:.6e} across {len(losses)} batches")

if __name__ == "__main__":
    main()

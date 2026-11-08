import argparse
import math
from pathlib import Path

import torch

from fno.datasets import generators

# Finds the data directory
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def main():

    """Generate training potentials and/or a labeled testing set."""

    parser = argparse.ArgumentParser(description=main.__doc__)

    # All the arguments needed for the dataset creation
    parser.add_argument(
        "-t", "--trainset", nargs="?", type=Path,
        const=DATA_DIR / "trainingset.pt",
        help="Training dataset path (default with -t: data/trainingset.pt).",
    )
    parser.add_argument(
        "-T", "--testset", nargs="?", type=Path,
        const=DATA_DIR / "testingset.pt",
        help="Testing dataset path (default with -T: data/testingset.pt).",
    )
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("-N", type=int, required=True, help="Number of grid points.")
    parser.add_argument(
        "-A", type=float, default=1.0,
        help="Half-width of the testing interval [-A, A] (default: 1).",
    )
    parser.add_argument(
        "-s", "--settype", choices=["mixed", "poly", "smooth", "well"],
        default="mixed", help="Family of random potentials (default: mixed).",
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Use CUDA if available.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Replace existing files.")
    args = parser.parse_args()

    # Checks that provided arguments work
    if args.trainset is None and args.testset is None:
        parser.error("Specify -t, -T, or both.")
    if args.num_batches < 1 or args.batch_size < 1 or args.N < 4:
        parser.error("Batch counts and sizes must be positive, and N must be at least 4.")
    if args.testset is not None and (not math.isfinite(args.A) or args.A <= 0):
        parser.error("A must be finite and positive.")
    if args.testset is not None and args.trainset is not None:
        if args.testset.resolve() == args.trainset.resolve():
            parser.error("Training and testing datasets need different output paths.")

    for path in (args.trainset, args.testset):
        if path is not None and path.exists() and not args.overwrite:
            parser.error(f"{path} already exists. Use -o to overwrite it.")

    # Loads device (either cuda or cpu)
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Both datasets use the same grid size and potential family, so options works for both
    options = {
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "N": args.N,
        "device": device,
        "type": args.settype,
    }

    # generates trainingset
    if args.trainset is not None:
        args.trainset.parent.mkdir(parents=True, exist_ok=True)
        generators.generate_training_set(args.trainset, **options)
        print(f"Saved training set to {args.trainset}")

    # Generates testingset
    if args.testset is not None:
        args.testset.parent.mkdir(parents=True, exist_ok=True)
        generators.generate_testing_set(args.testset, A=args.A, **options)
        print(f"Saved testing set to {args.testset}")


if __name__ == "__main__":
    main()

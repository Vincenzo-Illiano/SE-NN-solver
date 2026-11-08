import argparse
from pathlib import Path

from fno.models.fno1d import fno1d

def main():
    """Create and save an initialized model."""

    parser = argparse.ArgumentParser(description=main.__doc__)

    parser.add_argument(
        "-m", "--model", choices=["fno1d"], required=True,
        help="Type of model to create.",
    )

    parser.add_argument(
        "-p", "--path", type=Path, required=True,
        help="Path where the model checkpoint will be saved.",
    )

    # The branch currently supports one model type.
    fno1d.set_create_arguments(parser)
    args = parser.parse_args()

    try:
        model = fno1d.from_args(args)
    except ValueError as error:
        parser.error(str(error))

    args.path.parent.mkdir(parents=True, exist_ok=True)
    model.saveModel(args.path)

if __name__ == "__main__":
    main()

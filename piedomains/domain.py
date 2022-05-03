import sys
import argparse

from .pydomain import Pydomain

pred_shalla_cat = Pydomain.pred_shalla_cat


def main(argv=sys.argv[1:]):
    title = "Predict religion based on name"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--input", default=None, help="name")
    args = parser.parse_args(argv)
    print(args)
    if not args.input or not args.type:
        return -1

    output = pred_shalla_cat(args.input)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())

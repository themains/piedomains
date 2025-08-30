import sys
import argparse

from .piedomain import Piedomain

pred_shalla_cat = Piedomain.pred_shalla_cat
pred_shalla_cat_with_text = Piedomain.pred_shalla_cat_with_text
pred_shalla_cat_with_images = Piedomain.pred_shalla_cat_with_images

"""
Console script for piedomains.
"""


def main(argv=sys.argv[1:]):
    title = "Predict the category of the domain using the content of the domain and the screenshot of the homepage"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--input", default=None, help="Domain name to classify")
    args = parser.parse_args(argv)
    print(args)
    if not args.input:
        return -1

    output = pred_shalla_cat([args.input])
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())

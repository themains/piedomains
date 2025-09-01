import sys
import argparse

from .piedomain import Piedomain
from .archive_support import (
    pred_shalla_cat_archive,
    pred_shalla_cat_with_text_archive, 
    pred_shalla_cat_with_images_archive
)

pred_shalla_cat = Piedomain.pred_shalla_cat
pred_shalla_cat_with_text = Piedomain.pred_shalla_cat_with_text
pred_shalla_cat_with_images = Piedomain.pred_shalla_cat_with_images

"""
Console script for piedomains.
"""


def main(argv=sys.argv[1:]):
    title = "Predict the category of URLs or domains using content and homepage screenshots"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--input", default=None, help="URL or domain name to classify (e.g., 'example.com' or 'https://example.com/page')")
    args = parser.parse_args(argv)
    print(args)
    if not args.input:
        return -1

    output = pred_shalla_cat([args.input])
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())

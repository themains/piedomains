#!/usr/bin/env python3
"""
Screenshot capture script using modern PlaywrightFetcher.
Updated from legacy Selenium implementation for v0.5.0.
"""

import os
from pathlib import Path

import pandas as pd
from PIL import Image

from ..fetchers import PlaywrightFetcher


def main():
    """Main screenshot capture function."""
    df = pd.read_csv(
        "fulldomain_min_greater_than_5_words_v3.csv.gz", usecols=["full_domain"]
    )

    os.makedirs("png", exist_ok=True)

    with PlaywrightFetcher() as fetcher:
        for i, r in df.iterrows():
            fn = f"png/{i}.png"
            domain = r.full_domain
            url = f"http://{domain}"

            if not os.path.exists(fn):
                print(i, url)
                try:
                    # Use PlaywrightFetcher for screenshot capture
                    success, error = fetcher.fetch_screenshot(domain, "png")

                    if success:
                        # Move screenshot to desired location
                        source_path = Path("png") / f"{domain}.png"
                        if source_path.exists():
                            source_path.rename(fn)

                            # Convert PNG to JPG as in original script
                            img_png = Image.open(fn)
                            img_png = img_png.convert("RGB")  # Ensure RGB for JPG
                            img_png.save(fn.replace(".png", ".jpg"))

                            # Remove PNG image
                            os.unlink(fn)
                    else:
                        print("ERROR:", i, url, error)

                except Exception as e:
                    print("ERROR:", i, url, e)


if __name__ == "__main__":
    main()

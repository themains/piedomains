# -*- coding: utf-8 -*-

import os
import zipfile
import requests


REPO_BASE_URL = (
    os.environ.get("NEWPYDOMAINS_MODEL_URL")
    or "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBc1lTMFItTlcwcHJsdzhFTklFVGRyNDZkY1AzP2U9RXBLMTVs/root/content"
)


def download_file(url, target):
    status = True
    try:
        # download the file
        r = requests.get(url, allow_redirects=True)
        open(f"{target}/saved_model.zip", "wb").write(r.content)
        # unzip
        with zipfile.ZipFile(f"{target}/saved_model.zip", "r") as zip_ref:
            zip_ref.extractall(target)
        # remove zip file
        os.remove(f"{target}/saved_model.zip")
    except Exception as exe:
        print(f"Not able to download models {exe}")
        status = False
    return status

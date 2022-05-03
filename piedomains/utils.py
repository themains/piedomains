import os
import tarfile
import requests


REPO_BASE_URL = os.environ.get("PIEDOMAINS_MODEL_URL") or "https://dataverse.harvard.edu/api/access/datafile/6276339"


def download_file(url, target, file_name):
    status = True
    try:
        # download the file
        r = requests.get(url, allow_redirects=True)
        open(f"{target}/{file_name}", "wb").write(r.content)
        # untar
        with tarfile.open(f"{target}/{file_name}", "r:gz") as tar_ref:
            tar_ref.extractall(target)
        # remove zip file
        os.remove(f"{target}/{file_name}")
    except Exception as exe:
        print(f"Not able to download models {exe}")
        status = False
    return status

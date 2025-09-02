import os
import tarfile
import requests


REPO_BASE_URL = os.environ.get("PIEDOMAINS_MODEL_URL") or "https://dataverse.harvard.edu/api/access/datafile/7081895"


def download_file(url: str, target: str, file_name: str) -> bool:
    """
    Download file from the server

    Args:
        url (str): url of the resource
        target (str): target resource
        file_name (str): name of the filename to fetch
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    status = True
    try:
        # download the file
        r = requests.get(url, allow_redirects=True, timeout=10)
        open(f"{target}/{file_name}", "wb").write(r.content)
        # untar
        with tarfile.open(f"{target}/{file_name}", "r:gz") as tar_ref:
            safe_extract(tar_ref, target)
        # remove zip file
        os.remove(f"{target}/{file_name}")
    except Exception as exe:
        print(f"Not able to download models {exe}")
        status = False
    return status


def is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def safe_extract(tar, path: str = ".", members=None, *, numeric_owner: bool = False) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Failed Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)

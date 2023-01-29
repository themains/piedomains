import os
import tarfile
import requests


REPO_BASE_URL = os.environ.get("PIEDOMAINS_MODEL_URL") or "https://dataverse.harvard.edu/api/access/datafile/6908064"


def download_file(url, target, file_name):
    status = True
    try:
        # download the file
        r = requests.get(url, allow_redirects=True)
        open(f"{target}/{file_name}", "wb").write(r.content)
        # untar
        with tarfile.open(f"{target}/{file_name}", "r:gz") as tar_ref:

            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar_ref, target)
        # remove zip file
        os.remove(f"{target}/{file_name}")
    except Exception as exe:
        print(f"Not able to download models {exe}")
        status = False
    return status

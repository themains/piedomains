import os
from pkg_resources import resource_filename
from .utils import download_file, REPO_BASE_URL
from .logging import get_logger

logger = get_logger()

"""
Base class for all models
"""


class Base(object):
    MODELFN = None

    """
    Load model data from the server
    """

    @classmethod
    def load_model_data(cls, file_name: str, latest: bool = False) -> str:
        model_path = ""
        if cls.MODELFN:
            print(f"model fn {cls.MODELFN}")
            model_fn = resource_filename(__name__, cls.MODELFN)
            print(model_fn)
            if not os.path.exists(model_fn):
                os.makedirs(model_fn)
            if not os.path.exists(f"{model_fn}/saved_model") or latest:
                print(f"Downloading model data from the server (this is done only first time) ({model_fn})...")
                if not download_file(REPO_BASE_URL, f"{model_fn}", file_name):
                    logger.error("ERROR: Cannot download model data file")
            else:
                logger.debug(f"Using model data from {model_fn}...")
            model_path = model_fn

        return model_path

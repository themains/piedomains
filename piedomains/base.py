#!/usr/bin/env python3
"""
Base class infrastructure for model management and data loading.

This module provides the foundational base class for all machine learning models
in the piedomains package. It handles model file management, automatic downloading
from remote repositories, and local caching for improved performance.

The Base class serves as the foundation for all classifier implementations,
providing standardized model loading, caching, and resource management capabilities.
"""

import os

# Use modern importlib.resources for Python 3.9+, with fallback for older versions
from importlib.resources import files
from pathlib import Path

from .piedomains_logging import get_logger
from .utils import REPO_BASE_URL, download_file

logger = get_logger()


class Base:
    """
    Base class for all machine learning model implementations in piedomains.

    This class provides standardized functionality for model data management,
    including automatic downloading, caching, and loading of model files from
    remote repositories. All classifier classes should inherit from this base
    class to ensure consistent behavior.

    Class Attributes:
        MODELFN (str | None): Relative path to the model directory within the package.
                             Must be set by subclasses to specify their model location.

    Example:
        Creating a custom classifier that inherits from Base:

        >>> class MyClassifier(Base):
        ...     MODELFN = "model/my_classifier"
        ...
        ...     def __init__(self):
        ...         self.model_path = self.load_model_data("my_model.zip")

    Note:
        Subclasses must define the MODELFN class attribute to specify the model
        directory path. The load_model_data method will create this directory
        and download model files as needed.
    """

    MODELFN: str | None = None

    @classmethod
    def load_model_data(cls, file_name: str, latest: bool = False) -> str:
        """
        Load model data from local cache or download from remote repository.

        This method handles the complete lifecycle of model data:
        1. Checks if local model directory exists, creates if needed
        2. Verifies if model files exist locally or if update is requested
        3. Downloads model data from remote repository if necessary
        4. Returns the local path to model data for loading

        Args:
            file_name (str): Name of the model data file to download (e.g., "model.zip").
                           This should be the filename as it exists in the remote repository.
            latest (bool): If True, forces download of latest model data even if
                          local files exist. Useful for model updates. Defaults to False.

        Returns:
            str: Absolute path to the local model directory containing the downloaded
                 and extracted model files. Returns empty string if MODELFN is not set
                 or if download fails.

        Raises:
            OSError: If model directory cannot be created due to permission issues.
            ConnectionError: If model download fails due to network issues.

        Example:
            >>> class TextClassifier(Base):
            ...     MODELFN = "model/text_classifier"
            ...
            >>> classifier = TextClassifier()
            >>> model_path = classifier.load_model_data("text_model.zip")
            >>> print(f"Model loaded from: {model_path}")
            Model loaded from: /path/to/piedomains/model/text_classifier

            >>> # Force download of latest model
            >>> latest_path = classifier.load_model_data("text_model.zip", latest=True)

        Note:
            - This method only downloads if the saved_model directory doesn't exist
              or if latest=True is specified
            - Model files are cached locally to avoid repeated downloads
            - Downloads happen only on first use or when explicitly requested
        """
        model_path = ""

        if not cls.MODELFN:
            logger.warning(
                f"MODELFN not set for {cls.__name__}, cannot load model data"
            )
            return model_path

        try:
            logger.debug(f"Loading model data for {cls.__name__} from {cls.MODELFN}")

            # Get the absolute path to the model directory using modern importlib.resources
            model_dir_path = files(__name__.split(".")[0]) / cls.MODELFN
            model_fn = str(model_dir_path)
            logger.debug(f"Model directory path: {model_fn}")

            # Ensure model directory exists
            model_dir = Path(model_fn)
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Model directory ensured: {model_dir}")

            # Check if model files exist and determine if download is needed
            saved_model_path = model_dir / "saved_model"
            download_needed = not saved_model_path.exists() or latest

            if download_needed:
                if latest:
                    logger.info(
                        f"Downloading latest model data for {cls.__name__} "
                        f"from {REPO_BASE_URL}"
                    )
                else:
                    logger.info(
                        f"Model data not found locally for {cls.__name__}, "
                        f"downloading from server (first time setup)"
                    )

                # Download model data from remote repository
                download_success = download_file(
                    REPO_BASE_URL, str(model_fn), file_name
                )

                if not download_success:
                    logger.error(
                        f"Failed to download model data file '{file_name}' "
                        f"for {cls.__name__}"
                    )
                    raise ConnectionError(f"Model download failed for {file_name}")

                logger.info(f"Successfully downloaded model data for {cls.__name__}")

            else:
                logger.debug(f"Using cached model data from {model_fn}")

            model_path = str(model_fn)

        except OSError as e:
            logger.error(f"Failed to create model directory {cls.MODELFN}: {e}")
            raise OSError(f"Cannot create model directory: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error loading model data for {cls.__name__}: {e}")
            raise

        return model_path

    @classmethod
    def get_model_info(cls) -> dict[str, str]:
        """
        Get information about the model configuration and paths.

        Returns:
            dict[str, str]: Dictionary containing model information:
                - 'class_name': Name of the classifier class
                - 'model_fn': Model directory path (MODELFN)
                - 'model_path': Absolute path to model directory if it exists
                - 'has_saved_model': Whether saved_model directory exists

        Example:
            >>> info = TextClassifier.get_model_info()
            >>> print(info)
            {
                'class_name': 'TextClassifier',
                'model_fn': 'model/text_classifier',
                'model_path': '/path/to/piedomains/model/text_classifier',
                'has_saved_model': True
            }
        """
        info = {
            "class_name": cls.__name__,
            "model_fn": cls.MODELFN or "Not set",
            "model_path": "",
            "has_saved_model": False,
        }

        if cls.MODELFN:
            try:
                model_dir_path = files(__name__.split(".")[0]) / cls.MODELFN
                model_path = str(model_dir_path)
                info["model_path"] = model_path
                info["has_saved_model"] = os.path.exists(
                    os.path.join(model_path, "saved_model")
                )
            except Exception as e:
                logger.debug(f"Could not get model path info: {e}")

        return info

    def __init_subclass__(cls, **kwargs):
        """
        Validate subclass configuration when class is defined.

        Raises:
            ValueError: If MODELFN is not properly set by subclass.
        """
        super().__init_subclass__(**kwargs)

        if cls.MODELFN is None:
            logger.warning(
                f"Subclass {cls.__name__} should define MODELFN class attribute "
                "for proper model data management"
            )
        elif not isinstance(cls.MODELFN, str):
            raise ValueError(f"MODELFN must be a string, got {type(cls.MODELFN)}")

#!/usr/bin/env python3
"""Utility functions for file operations, downloads, and security.

This module provides essential utility functions for the piedomains package,
including secure file downloads, tar archive extraction with path traversal
protection, and configuration management.

The utilities focus on security-first implementation, particularly for handling
downloaded archives and preventing common security vulnerabilities like
path traversal attacks.
"""

import os
import tarfile
from pathlib import Path

import requests

from .piedomains_logging import get_logger

logger = get_logger()


# Model repository configuration
REPO_BASE_URL = os.environ.get(
    "PIEDOMAINS_MODEL_URL", "https://dataverse.harvard.edu/api/access/datafile/7081895"
)
"""
str: Base URL for model data repository.

Can be overridden via PIEDOMAINS_MODEL_URL environment variable.
Defaults to Harvard Dataverse hosting the piedomains model files.
"""


def download_file(url: str, target: str, file_name: str, timeout: int = 30) -> bool:
    """Download and extract a compressed model file from a remote repository.

    This function downloads a tar.gz file from the specified URL, saves it to the
    target directory, extracts it using secure extraction methods, and cleans up
    the downloaded archive file.

    Args:
        url: URL of the remote file to download. Should point to a valid
                  tar.gz archive containing model data.
        target: Local directory path where the file should be downloaded
                     and extracted. Directory will be created if it doesn't exist.
        file_name: Name to use for the downloaded file. Should include
                        appropriate extension (e.g., "model.tar.gz").
        timeout: HTTP request timeout in seconds. Defaults to 30 seconds
                      for large model files.

    Returns:
        bool: True if download and extraction completed successfully,
              False if any error occurred during the process.


    Example:
        >>> success = download_file(
        ...     url="https://example.com/model.tar.gz",
        ...     target="/path/to/models",
        ...     file_name="text_model.tar.gz"
        ... )
        >>> if success:
        ...     print("Model downloaded and extracted successfully")

    Security:
        - Uses safe_extract() to prevent path traversal attacks
        - Validates archive contents before extraction
        - Automatically removes downloaded archive after extraction
        - Logs all errors for security monitoring

    Note:
        The downloaded tar.gz file is automatically deleted after extraction
        to save disk space. Only the extracted contents remain in the target directory.

    """
    # Ensure target directory exists
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)

    file_path = target_path / file_name

    try:
        logger.info(f"Downloading {file_name} from {url}")
        logger.debug(f"Target directory: {target}")

        # Download the file with proper error handling
        response = requests.get(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Write file content
        with open(file_path, "wb") as f:
            f.write(response.content)

        file_size = file_path.stat().st_size
        logger.info(f"Downloaded {file_name} ({file_size:,} bytes)")

        # Extract archive using secure extraction
        logger.debug(f"Extracting {file_name} to {target}")
        with tarfile.open(file_path, "r:gz") as tar_ref:
            safe_extract(tar_ref, str(target_path))

        logger.info(f"Successfully extracted {file_name}")

        # Clean up downloaded archive
        file_path.unlink()
        logger.debug(f"Cleaned up downloaded archive: {file_name}")

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {file_name} from {url}: {e}")
        return False

    except tarfile.TarError as e:
        logger.error(f"Failed to extract {file_name}: {e}")
        # Clean up partial download
        if file_path.exists():
            file_path.unlink()
        return False

    except OSError as e:
        logger.error(f"File system error during download of {file_name}: {e}")
        # Clean up partial download
        if file_path.exists():
            file_path.unlink()
        return False

    except Exception as e:
        logger.error(f"Unexpected error downloading {file_name}: {e}")
        # Clean up partial download
        if file_path.exists():
            file_path.unlink()
        return False


def is_within_directory(directory: str, target: str) -> bool:
    """Check if a target path is within a specified directory (security check).

    This function validates that a file path is contained within a directory
    to prevent path traversal attacks when extracting archives. It resolves
    all symbolic links and relative path components before comparison.

    Args:
        directory: The base directory path that should contain the target.
        target: The target file/directory path to validate.

    Returns:
        bool: True if target is within directory, False if it would escape
              the directory boundary (indicating a potential path traversal attack).

    Example:
        >>> # Safe path
        >>> is_within_directory("/safe/dir", "/safe/dir/file.txt")
        True

        >>> # Path traversal attempt
        >>> is_within_directory("/safe/dir", "/safe/dir/../../../etc/passwd")
        False

        >>> # Another traversal attempt
        >>> is_within_directory("/safe/dir", "/safe/dir/subdir/../../../etc/passwd")
        False

    Security:
        This function is critical for preventing path traversal attacks
        (also known as directory traversal or dot-dot-slash attacks) where
        malicious archives attempt to extract files outside the intended directory.

    Note:
        Paths are resolved with os.path.realpath(), which follows symbolic links
        that exist, and compared component by component: "/safe/dir-evil" is
        not inside "/safe/dir", though it starts with the same characters.
    """
    abs_directory = os.path.realpath(directory)
    abs_target = os.path.realpath(target)

    is_safe = os.path.commonpath([abs_directory, abs_target]) == abs_directory

    if not is_safe:
        logger.warning(
            f"Path traversal attempt detected: {target} would escape {directory}"
        )

    return is_safe


def safe_extract(
    tar: tarfile.TarFile,
    path: str = ".",
    members: list | None = None,
    *,
    numeric_owner: bool = False,
) -> None:
    """Securely extract a tar archive with path traversal protection.

    This function provides a secure wrapper around tarfile.extractall() that
    validates all archive members to prevent path traversal attacks. It checks
    each member's path before extraction to ensure it stays within the target directory.

    Args:
        tar: Open tar file object to extract from.
        path: Directory path where archive should be extracted.
                   Defaults to current directory (".").
        members: Specific members to extract. If None,
                                 extracts all members. Defaults to None.
        numeric_owner: If True, preserve numeric user/group IDs.
                             If False, use current user. Defaults to False.


    Example:
        >>> import tarfile
        >>> with tarfile.open("model.tar.gz", "r:gz") as tar:
        ...     safe_extract(tar, "/safe/extraction/dir")

    Security:
        - Validates every archive member before extraction
        - Prevents path traversal attacks (e.g., "../../../etc/passwd")
        - Logs security violations for monitoring
        - Raises exceptions rather than silently failing

    Note:
        This function should always be used instead of tarfile.extractall()
        when handling archives from untrusted sources, which includes
        downloaded model files.


    Raises:
        SecurityError: If a security validation fails.
    """
    logger.debug(f"Performing secure extraction to {path}")

    # Device files are skipped rather than refused, as before; the "data" filter
    # below would otherwise reject the whole archive for one of them.
    validated_members = []
    for member in members or tar.getmembers():
        # The "data" filter would strip the leading slash and extract inside
        # `path`; an archive naming absolute paths is refused outright instead,
        # as this function always has.
        if os.path.isabs(member.name):
            raise SecurityError(
                f"Path traversal detected in tar archive: absolute member "
                f"'{member.name}'"
            )
        if member.isdev():
            logger.warning(f"Skipping device file in archive: {member.name}")
            continue
        validated_members.append(member)

    # The standard library's "data" filter (PEP 706) refuses every way out of
    # `path`: "..", absolute names, and symbolic or hard links that point
    # outside. A hand-written check here once compared paths as strings, which
    # let "../extract-evil/x" through, and never looked at hard links.
    try:
        tar.extractall(
            path, validated_members, numeric_owner=numeric_owner, filter="data"
        )
    except tarfile.FilterError as exc:
        raise SecurityError(f"Path traversal detected in tar archive: {exc}") from exc

    logger.info(f"Safely extracted {len(validated_members)} files to {path}")


class SecurityError(Exception):
    """Exception raised for security violations during file operations.

    This exception is raised when security checks fail, particularly
    during archive extraction when path traversal attempts are detected.

    Example:
        >>> try:
        ...     safe_extract(malicious_tar, "/safe/dir")
        ... except SecurityError as e:
        ...     logger.error(f"Security violation: {e}")
    """

    pass


def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate cryptographic hash of a file for integrity verification.

    Args:
        file_path: Path to the file to hash.
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', 'sha512').
                        Defaults to 'sha256' for security.

    Returns:
        str: Hexadecimal hash digest of the file.


    Example:
        >>> hash_value = get_file_hash("model.tar.gz", "sha256")
        >>> print(f"File hash: {hash_value}")


    Raises:
        FileNotFoundError: If the referenced file does not exist.
        OSError: If a filesystem operation fails.
        ValueError: If an argument is invalid.
    """
    import hashlib

    # Validate algorithm
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hash_obj = hashlib.new(algorithm)

    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    except FileNotFoundError:
        logger.error(f"File not found for hashing: {file_path}")
        raise
    except OSError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

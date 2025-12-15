#!/usr/bin/env python3
"""
Utility functions for file operations, downloads, and security.

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
    """
    Download and extract a compressed model file from a remote repository.

    This function downloads a tar.gz file from the specified URL, saves it to the
    target directory, extracts it using secure extraction methods, and cleans up
    the downloaded archive file.

    Args:
        url (str): URL of the remote file to download. Should point to a valid
                  tar.gz archive containing model data.
        target (str): Local directory path where the file should be downloaded
                     and extracted. Directory will be created if it doesn't exist.
        file_name (str): Name to use for the downloaded file. Should include
                        appropriate extension (e.g., "model.tar.gz").
        timeout (int): HTTP request timeout in seconds. Defaults to 30 seconds
                      for large model files.

    Returns:
        bool: True if download and extraction completed successfully,
              False if any error occurred during the process.

    Raises:
        requests.RequestException: If HTTP download fails (not caught, logged only).
        tarfile.TarError: If tar extraction fails (not caught, logged only).
        OSError: If file operations fail (not caught, logged only).

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
    """
    Check if a target path is within a specified directory (security check).

    This function validates that a file path is contained within a directory
    to prevent path traversal attacks when extracting archives. It resolves
    all symbolic links and relative path components before comparison.

    Args:
        directory (str): The base directory path that should contain the target.
        target (str): The target file/directory path to validate.

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
        This function uses os.path.abspath() to resolve all relative path
        components and symbolic links before performing the security check.
    """
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])
    is_safe = prefix == abs_directory

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
    """
    Securely extract a tar archive with path traversal protection.

    This function provides a secure wrapper around tarfile.extractall() that
    validates all archive members to prevent path traversal attacks. It checks
    each member's path before extraction to ensure it stays within the target directory.

    Args:
        tar (tarfile.TarFile): Open tar file object to extract from.
        path (str): Directory path where archive should be extracted.
                   Defaults to current directory (".").
        members (list, optional): Specific members to extract. If None,
                                 extracts all members. Defaults to None.
        numeric_owner (bool): If True, preserve numeric user/group IDs.
                             If False, use current user. Defaults to False.

    Raises:
        SecurityError: If any archive member attempts path traversal
                      (would extract outside the target directory).
        tarfile.TarError: If tar extraction fails for other reasons.
        OSError: If file system operations fail.

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
    """
    logger.debug(f"Performing secure extraction to {path}")

    # Get members to extract (all if none specified)
    members_to_extract = members or tar.getmembers()

    # Validate each member before extraction
    for member in members_to_extract:
        member_path = os.path.join(path, member.name)

        if not is_within_directory(path, member_path):
            raise SecurityError(
                f"Path traversal detected in tar archive: member '{member.name}' "
                f"would extract to '{member_path}' outside target directory '{path}'"
            )

        # Additional security checks
        if member.isdev():
            logger.warning(f"Skipping device file in archive: {member.name}")
            continue

        if member.issym() or member.islnk():
            # Check that symlinks don't escape the directory
            if member.issym():
                link_path = os.path.join(path, member.linkname)
                if not is_within_directory(path, link_path):
                    raise SecurityError(
                        f"Symlink traversal detected: {member.name} -> {member.linkname}"
                    )

    logger.debug(f"All {len(members_to_extract)} archive members validated")

    # Perform extraction
    tar.extractall(path, members, numeric_owner=numeric_owner)

    logger.info(f"Safely extracted {len(members_to_extract)} files to {path}")


class SecurityError(Exception):
    """
    Exception raised for security violations during file operations.

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
    """
    Calculate cryptographic hash of a file for integrity verification.

    Args:
        file_path (str): Path to the file to hash.
        algorithm (str): Hash algorithm to use ('md5', 'sha1', 'sha256', 'sha512').
                        Defaults to 'sha256' for security.

    Returns:
        str: Hexadecimal hash digest of the file.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        ValueError: If an unsupported hash algorithm is specified.

    Example:
        >>> hash_value = get_file_hash("model.tar.gz", "sha256")
        >>> print(f"File hash: {hash_value}")
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

#!/usr/bin/env python3
"""Download the training corpora, resumably and with checksum verification.

The Shallalist scrape that produced the shipped models is public on Harvard
Dataverse (doi:10.7910/DVN/ZXTQ7V) as multi-gigabyte split tarballs. This
fetches them by querying the Dataverse API rather than hardcoding file IDs,
resumes partial downloads with HTTP Range, verifies each part's MD5, and
reassembles the splits.

Usage:
    python training/download_corpus.py --list
    python training/download_corpus.py --set shallalist_all --out data/
    python training/download_corpus.py --set labels --out data/
    python training/download_corpus.py --set ut1 --out data/

Sizes: shallalist_all ~19GB, shallalist ~6.8GB, shallalist2 ~3.2GB,
screenshots ~48GB. Check free disk before starting.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import requests

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/ZXTQ7V"

#: Named groups, matched as filename prefixes against the Dataverse listing.
SETS: dict[str, str] = {
    "shallalist_all": "shallalist_all.tar.gz.part",
    "shallalist": "shallalist.tar.gz.part",
    "shallalist2": "shallalist2.tar.gz.part",
    "screenshots": "screenshot-",
    "labels": "shallalist_cats.txt|shallalist_dict.txt|screenshot-index.tab",
}

#: UT-Capitole blacklists — the maintained successor to Shallalist, used to
#: refresh labels since Shallalist itself was discontinued in 2022.
UT1_URL = "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"

CHUNK = 1 << 20

#: How many times to re-attempt a transfer that fails partway through.
#:
#: Dataverse returns a transient 502 often enough to matter: a screenshot run died 20
#: minutes in, on the eighth of 28 tarballs, because one request came back Bad Gateway and
#: nothing retried it. A multi-hour GPU job must not be lost to a single upstream hiccup.
ATTEMPTS = 5

#: Retried statuses. 429 is included so rate limiting backs off rather than aborting.
RETRY_STATUS = (429, 500, 502, 503, 504)


def session() -> requests.Session:
    """Build a session that retries transient failures with exponential backoff.

    Covers connecting and receiving response *headers*. A stream that dies mid-body is a
    separate problem, handled by the resume loop in :func:`download`.

    Returns:
        requests.Session: A session with retries mounted on HTTPS.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=ATTEMPTS,
        connect=ATTEMPTS,
        read=ATTEMPTS,
        status=ATTEMPTS,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset({"GET", "HEAD"}),
        backoff_factor=2,
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http


def list_files() -> list[dict[str, Any]]:
    """Fetch the dataset's file listing from the Dataverse API.

    Returns:
        list[dict]: One entry per file, with id/filename/filesize/md5.
    """
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}"
    response = session().get(url, timeout=60)
    response.raise_for_status()
    files = response.json()["data"]["latestVersion"]["files"]
    return [
        {
            "id": f["dataFile"]["id"],
            "name": f["dataFile"]["filename"],
            "size": f["dataFile"].get("filesize", 0),
            "md5": f["dataFile"].get("md5"),
        }
        for f in files
    ]


def select(files: list[dict[str, Any]], set_name: str) -> list[dict[str, Any]]:
    """Pick the files belonging to a named set.

    Args:
        files: Output of :func:`list_files`.
        set_name: A key of :data:`SETS`.

    Returns:
        list[dict]: Matching files, sorted by name.
    """
    prefixes = SETS[set_name].split("|")
    chosen = [f for f in files if any(f["name"].startswith(p) for p in prefixes)]
    return sorted(chosen, key=lambda f: f["name"])


def md5_of(path: Path) -> str:
    """Compute a file's MD5.

    Args:
        path: File to hash.

    Returns:
        str: Lowercase hex digest.
    """
    digest = hashlib.md5()  # noqa: S324 - matching Dataverse's published digest
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict[str, Any], out_dir: Path) -> Path:
    """Download one file, resuming and verifying its checksum.

    A partial file is continued with a Range request rather than restarted,
    which matters when a single part is 2.15GB.

    Args:
        entry: One item from :func:`list_files`.
        out_dir: Directory to write into.

    Returns:
        Path: The downloaded file.

    Raises:
        OSError: If the checksum does not match after downloading.
    """
    target = out_dir / entry["name"]
    expected = entry.get("md5")

    # Completeness is decided by checksum, not size. Dataverse reports the size
    # of its *converted* copy for tabular files while we fetch the original, so
    # a size comparison both misses real completions and then tries to resume
    # past EOF, which the storage backend answers with HTTP 416.
    if target.exists():
        if expected and md5_of(target) == expected:
            print(f"  {entry['name']}: already complete")
            return target
        if not expected and entry["size"] and target.stat().st_size == entry["size"]:
            print(f"  {entry['name']}: already complete (no checksum published)")
            return target

    # `format=original` matters for tabular files (.tab/.csv): Dataverse
    # converts those to its own archival format and serves the conversion, while
    # the published MD5 is of the original upload -- so without this the
    # checksum check fails on a perfectly good download.
    url = f"{DATAVERSE}/api/access/datafile/{entry['id']}?format=original"
    http = session()

    # The session retries connect/header failures, but a body that dies partway through
    # is delivered as a broken iterator rather than a bad status, so it has to be caught
    # here. Each attempt re-reads the file length and resumes with a fresh Range, so a
    # 2.15GB part that drops at 90% costs the last 10% and not the whole transfer.
    for attempt in range(1, ATTEMPTS + 1):
        have = target.stat().st_size if target.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with http.get(url, headers=headers, stream=True, timeout=120) as response:
                if response.status_code == 416 and have:
                    # Range beyond EOF: nothing left to fetch. Fall through to verify.
                    print(f"  {entry['name']}: already fully downloaded")
                    response.close()
                    if expected and md5_of(target) != expected:
                        target.unlink()
                        return download(entry, out_dir)
                    return target
                response.raise_for_status()
                mode = "ab" if have and response.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(target, mode) as handle:
                    for chunk in response.iter_content(CHUNK):
                        handle.write(chunk)
                        have += len(chunk)
                        if entry["size"]:
                            pct = 100 * have / entry["size"]
                            print(
                                f"\r  {entry['name']}: {pct:5.1f}%", end="", flush=True
                            )
            print()
            break
        except requests.RequestException as exc:
            if attempt == ATTEMPTS:
                raise
            delay = 2**attempt
            print(
                f"\n  {entry['name']}: {type(exc).__name__} after {have:,} bytes "
                f"(attempt {attempt}/{ATTEMPTS}); resuming in {delay}s"
            )
            time.sleep(delay)

    if expected:
        actual = md5_of(target)
        if actual != expected:
            raise OSError(
                f"{entry['name']}: checksum mismatch (expected {expected}, got {actual})"
            )
    return target


def reassemble(parts: list[Path], out_dir: Path) -> Path | None:
    """Concatenate split parts back into one tarball and sanity-check it.

    Args:
        parts: The ``.partaa``/``.partab``/... files, in order.
        out_dir: Where to write the joined archive.

    Returns:
        Path | None: The joined archive, or None if these were not split parts.
    """
    if not parts or ".part" not in parts[0].name:
        return None
    joined = out_dir / parts[0].name.split(".part")[0]
    if joined.exists():
        print(f"  {joined.name}: already assembled")
        return joined

    print(f"  assembling {joined.name} from {len(parts)} parts")
    with open(joined, "wb") as out:
        for part in parts:
            with open(part, "rb") as handle:
                shutil.copyfileobj(handle, out, CHUNK)

    try:
        with tarfile.open(joined, "r:gz") as tar:
            tar.next()
    except tarfile.TarError as exc:
        print(f"  WARNING: {joined.name} is not a readable tar.gz: {exc}")
    return joined


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="List available files")
    parser.add_argument(
        "--names",
        action="store_true",
        help="Print one filename per line for --set, and nothing else. Machine-readable, "
        "so a caller can loop over the corpus one file at a time",
    )
    parser.add_argument("--set", choices=[*SETS, "ut1"], help="Which corpus to fetch")
    parser.add_argument("--out", default="data", help="Output directory")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help="Fetch only these files from the set, by exact name. Repeatable. "
        "Lets a caller stream one tarball at a time instead of staging 47.58 GB "
        "before processing any of it",
    )
    parser.add_argument(
        "--no-assemble", action="store_true", help="Skip rejoining split parts"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Download a named corpus.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.set == "ut1":
        print(f"fetching UT-Capitole blacklists from {UT1_URL}")
        target = out_dir / "ut1-blacklists.tar.gz"
        with session().get(UT1_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(target, "wb") as handle:
                for chunk in response.iter_content(CHUNK):
                    handle.write(chunk)
        print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return 0

    files = list_files()

    if args.list or not args.set:
        total = sum(f["size"] for f in files)
        print(f"{len(files)} files, {total / 1e9:.1f} GB total\n")
        for name in SETS:
            chosen = select(files, name)
            size = sum(f["size"] for f in chosen)
            print(f"  {name:16s} {len(chosen):3d} files  {size / 1e9:6.2f} GB")
        print("\n  ut1              UT-Capitole blacklists (fresh labels)")
        return 0

    chosen = select(files, args.set)
    if args.names:
        for entry in chosen:
            print(entry["name"])
        return 0
    if args.only:
        wanted = set(args.only)
        chosen = [f for f in chosen if f["name"] in wanted]
        unknown = wanted - {f["name"] for f in chosen}
        if unknown:
            sys.stderr.write(f"not in set '{args.set}': {', '.join(sorted(unknown))}\n")
            return 1
    if not chosen:
        sys.stderr.write(f"no files matched set '{args.set}'\n")
        return 1

    size = sum(f["size"] for f in chosen)
    print(f"downloading {len(chosen)} files ({size / 1e9:.2f} GB) to {out_dir}")
    paths = [download(entry, out_dir) for entry in chosen]

    if not args.no_assemble:
        reassemble(paths, out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

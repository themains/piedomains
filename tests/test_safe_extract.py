"""Archive extraction must not write outside its target, however the escape is spelled.

Each case builds a tar whose member would land outside ``extract/``, extracts it, and
checks two things: ``SecurityError`` is raised, and nothing was written beside the
target. The sibling ``extract-evil`` matters: it shares the string prefix ``extract``,
which a character-by-character comparison mistakes for being inside.
"""

import io
import os
import tarfile
from pathlib import Path

import pytest

from piedomains import utils


def _tar(path: Path, member: tarfile.TarInfo, data: bytes = b"") -> Path:
    with tarfile.open(path, "w") as tar:
        member.size = len(data) if member.isfile() else 0
        tar.addfile(member, io.BytesIO(data) if member.isfile() else None)
    return path


def _file(name: str) -> tarfile.TarInfo:
    return tarfile.TarInfo(name)


def _link(name: str, target: str, kind: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, target
    return info


ESCAPES = {
    "parent directory": _file("../escaped.txt"),
    "sibling sharing the prefix": _file("../extract-evil/escaped.txt"),
    "absolute path": _file("/tmp/piedomains-escaped.txt"),
    "symlink outside": _link("link", "../outside", tarfile.SYMTYPE),
    "hard link outside": _link("hard", "../outside", tarfile.LNKTYPE),
}


@pytest.mark.parametrize("member", ESCAPES.values(), ids=ESCAPES.keys())
def test_an_escaping_member_is_refused(member, tmp_path):
    (tmp_path / "outside").write_text("secret")
    target = tmp_path / "extract"
    target.mkdir()
    archive = _tar(tmp_path / "evil.tar", member, b"payload")
    before = sorted(p.name for p in tmp_path.iterdir())

    with tarfile.open(archive) as tar, pytest.raises(utils.SecurityError):
        utils.safe_extract(tar, str(target))

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert list(target.iterdir()) == []
    assert (tmp_path / "outside").read_text() == "secret"


def test_an_ordinary_archive_still_extracts(tmp_path):
    target = tmp_path / "extract"
    target.mkdir()
    with tarfile.open(tmp_path / "ok.tar", "w") as tar:
        for name, data in (
            ("model/weights.bin", b"\x00\x01"),
            ("model/config.json", b"{}"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar.addfile(_link("model/latest", "weights.bin", tarfile.SYMTYPE))

    with tarfile.open(tmp_path / "ok.tar") as tar:
        utils.safe_extract(tar, str(target))

    assert (target / "model" / "weights.bin").read_bytes() == b"\x00\x01"
    assert os.readlink(target / "model" / "latest") == "weights.bin"


def test_a_device_file_is_skipped_not_extracted(tmp_path):
    target = tmp_path / "extract"
    target.mkdir()
    device = tarfile.TarInfo("dev/null")
    device.type = tarfile.CHRTYPE
    with tarfile.open(tmp_path / "dev.tar", "w") as tar:
        tar.addfile(device)
        info = tarfile.TarInfo("kept.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"ok"))

    with tarfile.open(tmp_path / "dev.tar") as tar:
        utils.safe_extract(tar, str(target))

    assert (target / "kept.txt").read_bytes() == b"ok"
    assert not (target / "dev").exists()


@pytest.mark.parametrize(
    ("directory", "target", "inside"),
    [
        ("/safe/dir", "/safe/dir/file.txt", True),
        ("/safe/dir", "/safe/dir", True),
        ("/safe/dir", "/safe/dir/../../etc/passwd", False),
        ("/safe/dir", "/safe/dir-evil/file.txt", False),  # shares the string prefix
        ("/safe/dir", "/safe/directory/file.txt", False),
    ],
)
def test_is_within_directory_compares_path_components(directory, target, inside):
    assert utils.is_within_directory(directory, target) is inside

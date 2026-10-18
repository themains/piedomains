"""The screenshot staging both Kaggle kernels run, with every external command faked.

The real job streams 47.58 GB from Dataverse, so these tests pin what it asks for and
how it reacts: which commands, in what order, what happens when a tarball fails, and
when it gives up.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from piedomains.training import stage_images as stage

TARBALLS = ["a.tar.gz", "b.tar.gz", "c.tar.gz"]


def _fake_commands(monkeypatch, temp: Path, failing: set[str]) -> list[list[str]]:
    """Record every command; ``--only <name>`` downloads fail for names in ``failing``.

    Returns:
        list[list[str]]: The commands, with the temp dir and interpreter abbreviated.
    """
    log: list[list[str]] = []

    def short(cmd: list[str]) -> list[str]:
        return ["PY" if c == sys.executable else c.replace(str(temp), "T") for c in cmd]

    def run(cmd: list[str]) -> None:
        log.append(short(cmd))
        if "--only" in cmd:
            name = cmd[cmd.index("--only") + 1]
            if name in failing:
                raise subprocess.CalledProcessError(1, cmd)
            (Path(cmd[cmd.index("--out") + 1]) / name).write_bytes(b"archive")

    def listing(cmd: list[str], **_: object) -> types.SimpleNamespace:
        log.append(short(cmd))
        return types.SimpleNamespace(
            stdout="\n".join(TARBALLS), stderr="", returncode=0
        )

    monkeypatch.setattr(stage, "_run", run)
    monkeypatch.setattr(
        stage,
        "subprocess",
        types.SimpleNamespace(
            run=listing, CalledProcessError=subprocess.CalledProcessError
        ),
    )
    return log


def test_each_tarball_is_fetched_resized_and_deleted(monkeypatch, tmp_path):
    log = _fake_commands(monkeypatch, tmp_path, failing=set())

    out = stage.stage_images(tmp_path, size=224, max_per_class=3000)

    assert out == tmp_path / "images-224"
    steps = [c[2] if c[1] == "-m" else c[1] for c in log]
    assert steps == [
        "piedomains.training.download_corpus",  # labels
        "-c",  # label cache
        "piedomains.training.download_corpus",  # --names listing
        *["piedomains.training.download_corpus", "piedomains.training.prepare_images"]
        * 3,
    ]
    resize = log[4]
    assert resize[resize.index("--size") + 1] == "224"
    assert resize[resize.index("--max-per-class") + 1] == "3000"
    assert "--append" in resize
    assert not list((tmp_path / "corpus").glob("*.tar.gz"))


def test_a_failed_tarball_is_skipped_and_reported(monkeypatch, tmp_path, capsys):
    log = _fake_commands(monkeypatch, tmp_path, failing={"b.tar.gz"})

    stage.stage_images(tmp_path, size=224, max_per_class=3000)

    fetched = [c[c.index("--only") + 1] for c in log if "--only" in c]
    assert fetched == TARBALLS
    assert sum("piedomains.training.prepare_images" in c for c in log) == 2
    assert "1/3 tarballs skipped: b.tar.gz" in capsys.readouterr().out


def test_every_tarball_failing_stops_the_run(monkeypatch, tmp_path):
    _fake_commands(monkeypatch, tmp_path, failing=set(TARBALLS))

    with pytest.raises(SystemExit, match="every tarball failed"):
        stage.stage_images(tmp_path, size=224, max_per_class=3000)


def test_an_existing_dataset_is_not_rebuilt(monkeypatch, tmp_path):
    log = _fake_commands(monkeypatch, tmp_path, failing=set())
    (tmp_path / "images-224").mkdir()
    (tmp_path / "images-224" / "labels.json").write_text("[]")

    assert stage.stage_images(tmp_path, size=224, max_per_class=3000) == (
        tmp_path / "images-224"
    )
    assert log == []

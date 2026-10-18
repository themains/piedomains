"""The GPU check every Kaggle kernel shares, as it runs inside a built notebook.

The kernels import ``kaggle/_bootstrap.py``; ``push_kernel.py`` pastes it in when it
builds the notebook, because the notebook runs the kernel as a single file. These tests
build each notebook's source, load it the way Kaggle runs it, and drive its GPU check
through faked probe results.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from piedomains.training.kaggle.push_kernel import build_source, inline_bootstrap, main

KAGGLE = Path(__file__).parents[1] / "src" / "piedomains" / "training" / "kaggle"
KERNELS = ["train_image_kaggle.py", "train_joint_kaggle.py", "train_text_kaggle.py"]


def _notebook_module(
    kernel: str, probe_codes: list[int], log: list
) -> types.ModuleType:
    """Load a kernel's built notebook source, with the probe and runner faked.

    Returns:
        types.ModuleType: The loaded notebook script.
    """
    source = build_source(None, KAGGLE / kernel)
    module = types.ModuleType("notebook")
    exec(compile(source, kernel, "exec"), module.__dict__)  # noqa: S102 - our own kernel
    codes = list(probe_codes)

    def probe(_cmd: list[str], **_: object) -> types.SimpleNamespace:
        log.append("probe")
        return types.SimpleNamespace(returncode=codes.pop(0), stdout="", stderr="")

    module.__dict__["subprocess"] = types.SimpleNamespace(
        run=probe, CalledProcessError=subprocess.CalledProcessError
    )
    return module


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_notebook_carries_the_check_instead_of_importing_it(kernel):
    source = build_source(None, KAGGLE / kernel)

    assert "import ensure_usable_gpu" not in source
    assert "def ensure_usable_gpu(" in source
    compile(source, kernel, "exec")


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    ("probe_codes", "outcome", "reinstalls"),
    [
        ([0], "returns", 0),  # the GPU works
        ([42, 0], "returns", 1),  # a P100: reinstall a build with sm_60, then it works
        ([42, 1], "exits", 1),  # still unusable after the reinstall
        ([1], "exits", 0),  # no GPU at all; reinstalling would not help
    ],
)
def test_gpu_check_inside_the_notebook(
    kernel, probe_codes, outcome, reinstalls, tmp_path
):
    log: list = []
    notebook = _notebook_module(kernel, probe_codes, log)

    def run(cmd: list[str]) -> None:
        log.append(cmd)

    if outcome == "returns":
        notebook.ensure_usable_gpu(tmp_path, run)
    else:
        with pytest.raises(SystemExit):
            notebook.ensure_usable_gpu(tmp_path, run)

    installs = [c for c in log if c != "probe"]
    assert len(installs) == reinstalls
    for cmd in installs:
        assert cmd[:3] == [sys.executable, "-m", "pip"]
        assert "--force-reinstall" in cmd
        assert f"torch=={notebook.PASCAL_TORCH}" in cmd
    assert log.count("probe") == len(probe_codes)
    assert (tmp_path / "gpu_probe.py").read_text() == notebook.GPU_PROBE


def test_a_kernel_without_the_import_is_refused(tmp_path):
    kernel = tmp_path / "train_other_kaggle.py"

    with pytest.raises(SystemExit, match=r"must import \._bootstrap exactly once"):
        inline_bootstrap("def main(): ...\n", kernel)


@pytest.mark.parametrize("kernel", KERNELS)
def test_print_gives_the_script_to_paste_by_hand(kernel, capsys):
    assert main(["--script", str(KAGGLE / kernel), "--print"]) == 0

    printed = capsys.readouterr().out
    assert printed.strip() == build_source(None, KAGGLE / kernel).strip()
    assert "import ensure_usable_gpu" not in printed

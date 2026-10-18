"""GPU check every Kaggle kernel runs before anything expensive.

Each kernel runs as a single file pasted into a notebook, before the repository is
cloned, so it cannot import this. ``push_kernel.py`` pastes everything below the
``INLINE`` marker in place of the kernel's ``from ._bootstrap import`` line, so the
kernels share one copy here and still run as one file there.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# INLINE: everything below is pasted into each notebook by push_kernel.py.

#: Last torch whose cu121 wheels still ship Pascal (sm_60) kernels, for when Kaggle hands
#: out a P100. Kaggle's own preinstalled build is cu128 and starts at sm_70.
PASCAL_TORCH = "2.6.0"
PASCAL_TORCHVISION = "0.21.0"

#: cu124, not cu121: the cu121 index stops at torch 2.5.1, which transformers
#: rejects. Three constraints have to hold at once -- transformers needs >= 2.6,
#: the P100 needs sm_60 kernels, and the wheel has to exist on the index.
PASCAL_INDEX = "https://download.pytorch.org/whl/cu124"

#: Probe run in a subprocess, because a GPU verdict cannot be revised inside a process
#: that has already initialized CUDA. Exit 42 means "device present but this torch has no
#: kernels for it" -- the failure that wasted a whole run.
GPU_PROBE = """
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    print("no CUDA device")
    sys.exit(1)
major, minor = torch.cuda.get_device_capability()
sm = f"sm_{major}{minor}"
arches = torch.cuda.get_arch_list()
print("device", torch.cuda.get_device_name(0), sm)
print("this torch builds for", arches)
if sm not in arches:
    print(f"INCOMPATIBLE: {sm} is not in the build")
    sys.exit(42)
# Arch lists can still lie; a real kernel launch is the only proof.
torch.zeros(8, device="cuda").add_(1).sum().item()
print("OK: a real kernel launched on", sm)
"""


def ensure_usable_gpu(scratch: Path, run: Callable[[list[str]], None]) -> None:
    """Prove the GPU can actually run a kernel, before anything expensive.

    Kaggle assigns either a T4 (sm_75) or a P100 (sm_60), and current PyTorch builds have
    dropped Pascal. A run got a P100, spent an hour on stage 1, then died instantly on
    ``no kernel image is available for execution on the device`` -- with no checkpoint and
    the whole session wasted. Checking first costs 30 seconds.

    A capability mismatch is repaired by installing a build that covers the device rather
    than giving up, since which GPU Kaggle hands out is not ours to choose.

    Args:
        scratch: Directory the probe script is written to.
        run: The kernel's command runner, used for the reinstall.

    Raises:
        SystemExit: If no GPU is present, or if it is still unusable after reinstalling.
            Fatal on purpose: CPU training on 51,138 images would not finish inside the
            session cap, so continuing would burn the run to no end.
    """
    probe = scratch / "gpu_probe.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(GPU_PROBE, encoding="utf-8")

    first = subprocess.run(  # noqa: S603 -- interpreter plus a file we just wrote
        [sys.executable, str(probe)], capture_output=True, text=True, check=False
    )
    print(first.stdout.strip() or first.stderr.strip()[:600])
    if first.returncode == 0:
        return
    if first.returncode != 42:
        raise SystemExit("no usable GPU; set the accelerator on this notebook to GPU")

    print(f"\ninstalling torch {PASCAL_TORCH} for this device's compute capability...")
    # Pinned and forced. An unpinned `pip install --index-url ... torch` is a no-op here:
    # pip sees the preinstalled 2.10.0+cu128 as already satisfying `torch` and changes
    # nothing, so the second probe reported the identical incompatible build.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "--no-cache-dir",
            "--index-url",
            PASCAL_INDEX,
            f"torch=={PASCAL_TORCH}",
            f"torchvision=={PASCAL_TORCHVISION}",
        ]
    )
    second = subprocess.run(  # noqa: S603 -- interpreter plus a file we just wrote
        [sys.executable, str(probe)], capture_output=True, text=True, check=False
    )
    print(second.stdout.strip() or second.stderr.strip()[:600])
    if second.returncode != 0:
        raise SystemExit(
            "GPU still unusable after reinstalling torch. Switch this notebook's "
            "accelerator to 'GPU T4 x2' (sm_75) and re-run; P100 is sm_60, which "
            "current torch builds no longer ship kernels for."
        )

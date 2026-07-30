"""Training and evaluation scripts, shipped with the package.

Every accuracy figure in the README is produced by something in here, and a number nobody
can re-run is a number taken on faith. So these install alongside the library rather than
living only in the repository: ``pip install piedomains`` is enough to reproduce any
published claim.

Each script is a module with a ``__main__`` guard::

    python -m piedomains.training.evaluate --method text
    python -m piedomains.training.train_text --data data/prepared --out models/text-v5

Fine-tuning additionally needs the ``train`` dependency group (``datasets``,
``accelerate``, ``torchvision``); inference-only dependencies are already required by the
library itself.

**This module imports nothing on purpose.** Pulling ``torch`` or ``transformers`` in here
would make every ``import piedomains.training`` pay for machinery a caller listing the
available scripts does not want, and would make the package's import cost depend on code
that only ever runs from a command line. The heavy imports stay inside the functions that
need them, which is the same pattern the library uses.
"""

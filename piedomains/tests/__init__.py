import sys
from contextlib import contextmanager
import pandas

try:
    from StringIO import StringIO
except Exception:
    from io import StringIO


@contextmanager
def capture(command, *args, **kwargs):
    out, sys.stdout = sys.stdout, StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out

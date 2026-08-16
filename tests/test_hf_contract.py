"""Live contract for immutable model repositories."""

import pytest

from piedomains.image import DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_REVISION
from piedomains.text import DEFAULT_TEXT_MODEL, DEFAULT_TEXT_REVISION


def test_default_revisions_are_immutable_commits() -> None:
    for revision in (DEFAULT_TEXT_REVISION, DEFAULT_IMAGE_REVISION):
        assert len(revision) == 40
        assert set(revision) <= set("0123456789abcdef")


@pytest.mark.live
@pytest.mark.parametrize(
    ("repo", "revision"),
    [
        (DEFAULT_TEXT_MODEL, DEFAULT_TEXT_REVISION),
        (DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_REVISION),
    ],
)
def test_pinned_model_revision_exists(repo: str, revision: str) -> None:
    from huggingface_hub import list_repo_files

    assert list_repo_files(repo, revision=revision)

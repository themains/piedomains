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


@pytest.mark.live
def test_pinned_fusion_matches_the_runtime_taxonomy() -> None:
    from piedomains.checkpoints import read_sidecar
    from piedomains.fusion import fuse_probabilities, load_fusion_weights

    text_labels = read_sidecar(
        DEFAULT_TEXT_MODEL, "labels.json", revision=DEFAULT_TEXT_REVISION
    )
    image_labels = read_sidecar(
        DEFAULT_IMAGE_MODEL, "labels.json", revision=DEFAULT_IMAGE_REVISION
    )
    weights = load_fusion_weights(DEFAULT_IMAGE_MODEL, revision=DEFAULT_IMAGE_REVISION)

    assert text_labels is not None
    assert image_labels is not None
    assert weights is not None
    assert weights.labels == tuple(text_labels)
    assert len(weights.text) in (1, len(text_labels))

    text = {label: 1 / len(text_labels) for label in text_labels}
    image = {label: 1 / len(image_labels) for label in image_labels}
    fused = fuse_probabilities(text, image, weights)
    assert tuple(fused) == tuple(text_labels)
    assert sum(fused.values()) == pytest.approx(1.0)

    for index, label in enumerate(text_labels):
        if label not in image:
            assert weights.weight_for(index) == 1.0

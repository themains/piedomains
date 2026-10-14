"""Offline relabeling preserves document joins, spend gates, and recovery."""

import json

import batchlane as bl
import pytest
import respx

from piedomains.training import relabel
from piedomains.training.relabel import BudgetExceededError, ask_batches

BASE = "https://api.openai.com/v1"
DOCS = [
    ("train", {"domain": "a.example", "text": "a pharmacy", "category": "drugs"}),
    ("val", {"domain": "b.example", "text": "a store", "category": "shopping"}),
]
PARAMS = {"model": "openai/gpt-4o-mini", "max_tokens": 300, "temperature": 0.0}


def run(tmp_path, pending=None, selected=None, **kwargs):
    return ask_batches(
        DOCS if pending is None else pending,
        DOCS if selected is None else selected,
        ["drugs", "shopping"],
        {},
        PARAMS,
        100,
        "prompt-id",
        batch_size=1,
        checkpoint=tmp_path / "run.batchlane.jsonl",
        budget_usd=kwargs.pop("budget_usd", 1),
        **kwargs,
    )


@pytest.fixture
def api(monkeypatch):
    import batchlane.cost

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(
        batchlane.cost,
        "estimate_cost",
        lambda *a: bl.CostEstimate(30, 600, 0.01, 0.02, "published"),
    )
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/files", name="upload").respond(200, json={"id": "in"})
        router.post(f"{BASE}/batches", name="submit").respond(200, json={"id": "job"})
        router.get(f"{BASE}/batches/job", name="poll").respond(
            200, json={"status": "completed", "output_file_id": "out"}
        )
        rows = []
        for index in [1, 0]:
            record = DOCS[index][1]
            answer = {
                "domain": record["domain"],
                "category": record["category"],
                "confidence": 0.9,
                "reasoning": "page text",
                "evidence": record["text"],
                "substance_type": "none",
            }
            rows.append(
                {
                    "custom_id": f"group-{index}",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "model": "gpt-4o-mini",
                            "choices": [{"message": {"content": json.dumps([answer])}}],
                            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                        },
                    },
                }
            )
        router.get(f"{BASE}/files/out/content", name="output").respond(
            200, text="\n".join(json.dumps(row) for row in rows)
        )
        yield router


def test_real_batch_transport_preserves_domains_splits_and_verdicts(api, tmp_path):
    collected = list(run(tmp_path))
    verdicts = {v.domain: v for group, _ in collected for v in group}
    assert set(verdicts) == {"a.example", "b.example"}
    assert verdicts["a.example"].split == "train"
    assert verdicts["b.example"].split == "val"
    assert verdicts["a.example"].status == "ok"
    payload = api["upload"].calls[0].request.content.decode()
    assert "a pharmacy" in payload and "a store" in payload


def test_partial_collection_resumes_exact_request_snapshot(api, tmp_path):
    first = run(tmp_path)
    collected, _ = next(first)
    first.close()
    remaining = [row for row in DOCS if row[1]["domain"] != collected[0].domain]
    resumed = list(run(tmp_path, pending=remaining))
    assert [v.domain for group, _ in resumed for v in group] == [
        remaining[0][1]["domain"]
    ]
    assert api["submit"].call_count == 1


def test_changed_document_refuses_existing_journal(api, tmp_path):
    list(run(tmp_path))
    changed = [(split, {**record, "text": "different"}) for split, record in DOCS]
    with pytest.raises(ValueError, match="documents differ"):
        list(run(tmp_path, selected=changed))
    assert api["submit"].call_count == 1


def test_budget_gate_runs_before_any_provider_call(api, tmp_path):
    with pytest.raises(BudgetExceededError, match="above"):
        list(run(tmp_path, budget_usd=0.001))
    assert not api.calls
    assert not (tmp_path / "run.batchlane.batchstate.json").exists()


def test_unpriced_model_refuses_submission(api, tmp_path, monkeypatch):
    import batchlane.cost

    monkeypatch.setattr(
        batchlane.cost,
        "estimate_cost",
        lambda *a: bl.CostEstimate(30, 600, None, None, "unknown"),
    )
    with pytest.raises(BudgetExceededError, match="cannot bound"):
        list(run(tmp_path))
    assert not api.calls


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1, 0])
def test_invalid_budget_refuses(api, tmp_path, budget):
    with pytest.raises(ValueError, match="positive"):
        list(run(tmp_path, budget_usd=budget))
    assert not api.calls


@pytest.fixture
def cli(tmp_path):
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "labels.json").write_text(json.dumps(["drugs", "shopping"]))
    for split in relabel.SPLITS:
        records = [r for _, r in DOCS if relabel.split_of(r["domain"]) == split]
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    return [
        "--data",
        str(data),
        "--verdicts",
        str(tmp_path / "results" / "verdicts.jsonl"),
        "--definitions",
        str(tmp_path / "no-definitions"),
        "--only-classes",
        "",
        "--provider",
        "openai",
        "--model",
        "gpt-4o-mini",
        "--batch",
        "--batch-size",
        "1",
        "--max-tokens",
        "300",
    ]


def verdicts_file(cli):
    from pathlib import Path

    return Path(cli[cli.index("--verdicts") + 1])


@pytest.mark.parametrize("dry_run", [False, True])
def test_completed_cli_cache_validates_documents(api, cli, dry_run):
    from pathlib import Path

    assert relabel.main(cli) == 0
    path = Path(cli[1]) / f"{relabel.split_of('a.example')}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["text"] = "a different page"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = len(api.calls)
    manifest = verdicts_file(cli).with_suffix(".manifest.json").read_bytes()
    with pytest.raises(ValueError, match="documents differ"):
        relabel.main(cli + (["--dry-run"] if dry_run else []))
    assert len(api.calls) == before
    assert verdicts_file(cli).with_suffix(".manifest.json").read_bytes() == manifest


@pytest.mark.parametrize(
    ("option", "value"),
    [("--model", "gpt-4o"), ("--temperature", "0.5"), ("--batch-size", "2")],
)
def test_completed_cli_cache_validates_configuration(api, cli, option, value):
    assert relabel.main(cli) == 0
    before = len(api.calls)
    with pytest.raises(ValueError, match="configuration differs"):
        relabel.main([*cli, option, value])
    assert len(api.calls) == before


def test_completed_cli_cache_needs_no_provider_or_pricing(api, cli, monkeypatch):
    assert relabel.main(cli) == 0
    before = len(api.calls)
    verdicts = verdicts_file(cli).read_bytes()

    def no_plan(*args, **kwargs):
        pytest.fail("Completed jobs must not be repriced")

    monkeypatch.setattr(bl, "plan", no_plan)
    monkeypatch.delenv("OPENAI_API_KEY")
    assert relabel.main(cli) == 0
    assert relabel.main([*cli, "--dry-run"]) == 0
    assert len(api.calls) == before
    assert verdicts_file(cli).read_bytes() == verdicts


def test_fresh_dry_run_creates_no_batch_files(api, cli):
    assert relabel.main([*cli, "--dry-run"]) == 0
    assert not api.calls
    assert not verdicts_file(cli).parent.exists()


@pytest.mark.parametrize("state", ["failed", "expired", "cancelled"])
def test_failed_provider_job_is_not_cached_as_complete(
    api, cli, state, tmp_path, capsys
):
    api["poll"].respond(200, json={"status": state})
    output = tmp_path / "corrected"
    assert relabel.main([*cli, "--out", str(output)]) == 1
    assert "Provider batch requests failed" in capsys.readouterr().err
    assert not verdicts_file(cli).exists()
    assert not output.exists()
    assert relabel.main(cli) == 1
    assert api["submit"].call_count == 1
    assert api["poll"].call_count == 2


def test_request_errors_retain_successes_and_allow_explicit_retry(api, cli, capsys):
    content = api["output"]
    responses = [json.loads(row) for row in content.return_value.text.splitlines()]
    responses[1] = {
        "custom_id": "group-0",
        "response": {"status_code": 500, "body": {"error": {"message": "overloaded"}}},
    }
    content.respond(200, text="\n".join(json.dumps(row) for row in responses))
    assert relabel.main(cli) == 1
    assert "overloaded" in capsys.readouterr().err
    path = verdicts_file(cli)
    saved = path.read_bytes()
    rows = [json.loads(row) for row in saved.splitlines()]
    assert [(row["domain"], row["status"]) for row in rows] == [("b.example", "ok")]
    assert relabel.main(cli) == 1
    assert path.read_bytes() == saved
    assert api["submit"].call_count == 1

    retry_path = path.with_name("retry.jsonl")
    retry_path.write_bytes(saved)
    answer = {
        "domain": "a.example",
        "category": "drugs",
        "confidence": 0.9,
        "reasoning": "page text",
        "evidence": "a pharmacy",
        "substance_type": "none",
    }
    content.respond(
        200,
        text=json.dumps(
            {
                "custom_id": "group-0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"content": json.dumps([answer])}}],
                    },
                },
            }
        ),
    )
    assert relabel.main([*cli, "--verdicts", str(retry_path)]) == 0
    assert api["submit"].call_count == 2
    payload = api["upload"].calls[-1].request.content.decode()
    assert "a pharmacy" in payload and "a store" not in payload
    assert len(retry_path.read_text().splitlines()) == 2


def test_cli_timeout_resumes_without_duplicate_submission(api, cli):
    poll = api["poll"]
    poll.respond(200, json={"status": "in_progress"})
    assert relabel.main([*cli, "--poll-timeout", "0"]) == 1
    assert not verdicts_file(cli).exists()
    poll.respond(200, json={"status": "completed", "output_file_id": "out"})
    assert relabel.main(cli) == 0
    assert api["submit"].call_count == 1
    assert len(verdicts_file(cli).read_text().splitlines()) == 2


def test_cli_interrupted_collection_does_not_append_saved_verdicts_again(
    api, cli, monkeypatch
):
    append = relabel.append_verdicts

    def interrupted(path, verdicts):
        append(path, verdicts)
        raise RuntimeError("interrupted after persisting a result")

    monkeypatch.setattr(relabel, "append_verdicts", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        relabel.main(cli)
    saved = verdicts_file(cli).read_bytes()
    assert len(saved.splitlines()) == 1
    monkeypatch.setattr(relabel, "append_verdicts", append)
    assert relabel.main(cli) == 0
    completed = verdicts_file(cli).read_bytes()
    assert completed.startswith(saved)
    assert len(completed.splitlines()) == 2
    assert api["submit"].call_count == 1


# LiteLLM's response schema uses ReadOnly; Pydantic warns that dicts remain mutable.
@pytest.mark.filterwarnings(
    "ignore:Item 'summary' on TypedDict class 'ChatCompletionReasoningItem' is using the `ReadOnly` qualifier.*:UserWarning:pydantic\\._internal\\._generate_schema"
)
def test_anthropic_batch_transport_decodes_native_responses(api, cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    base = "https://api.anthropic.com/v1/messages/batches"
    api.post(base).respond(200, json={"id": "anthro-job"})
    api.get(f"{base}/anthro-job").respond(
        200,
        json={
            "id": "anthro-job",
            "processing_status": "ended",
            "request_counts": {"succeeded": 2},
            "results_url": f"{base}/anthro-job/results",
        },
    )
    rows = []
    for index in [1, 0]:
        record = DOCS[index][1]
        answer = {
            "domain": record["domain"],
            "category": record["category"],
            "confidence": 0.9,
            "reasoning": "page text",
            "evidence": record["text"],
            "substance_type": "none",
        }
        rows.append(
            {
                "custom_id": f"group-{index}",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": f"message-{index}",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [{"type": "text", "text": json.dumps([answer])}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 20, "output_tokens": 10},
                    },
                },
            }
        )
    api.get(f"{base}/anthro-job/results").respond(
        200, text="\n".join(json.dumps(row) for row in rows)
    )
    assert (
        relabel.main(
            [
                *cli,
                "--provider",
                "anthropic",
                "--model",
                "claude-sonnet-4-20250514",
            ]
        )
        == 0
    )
    verdicts = [json.loads(row) for row in verdicts_file(cli).read_text().splitlines()]
    assert {row["domain"] for row in verdicts} == {
        record["domain"] for _, record in DOCS
    }
    assert all(row["status"] == "ok" for row in verdicts)

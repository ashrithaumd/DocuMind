"""Re-judging keeps each statement's verdict, one record per pass (no network:
the metric is a stub), and the stub's shape is held to the real pinned RAGAS."""
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import rejudge_faithfulness  # noqa: E402
import run_eval  # noqa: E402

SAMPLE = {"user_input": "q", "response": "a", "retrieved_contexts": ["c"], "reference": "r"}


class Verdict(BaseModel):
    statement: str
    reason: str
    verdict: int


class StubMetric:
    """Each pass takes the next item of `script`: a verdict list, or an
    exception to raise."""

    def __init__(self, script):
        self.script = iter(script)

    async def _create_statements(self, row, callbacks):
        self.current = next(self.script)
        if isinstance(self.current, Exception):
            raise self.current
        return SimpleNamespace(statements=[v.statement for v in self.current])

    async def _create_verdicts(self, row, statements, callbacks):
        return SimpleNamespace(statements=self.current)

    def _compute_score(self, verdicts):
        return sum(v.verdict for v in verdicts.statements) / len(verdicts.statements)


def run(script, tmp_path, passes=None):
    out = tmp_path / "out.jsonl"
    counter = SimpleNamespace(requests={}, errors=[])
    rejudge_faithfulness.rejudge(SAMPLE, StubMetric(script), counter, passes or len(script), out,
                                 {"judge_model": "m"})
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_every_pass_keeps_each_statement_and_its_verdict(tmp_path):
    supported = Verdict(statement="s1", reason="in context", verdict=1)
    unsupported = Verdict(statement="s2", reason="not in context", verdict=0)

    records = run([[supported, unsupported], [supported, supported]], tmp_path)

    assert [r["score"] for r in records] == [0.5, 1.0]
    assert records[0]["statements"] == ["s1", "s2"]
    assert records[0]["verdicts"][1] == {"statement": "s2", "reason": "not in context", "verdict": 0}
    assert records[0]["response"] == "a" and records[0]["judge_model"] == "m"


def test_failed_pass_is_recorded_and_the_next_one_runs(tmp_path):
    supported = Verdict(statement="s1", reason="in context", verdict=1)

    records = run([TimeoutError("slow"), [supported]], tmp_path)

    assert records[0]["score"] is None and "slow" in records[0]["error"]
    assert records[1]["score"] == 1.0


def test_daily_quota_stops_the_run_after_saving_that_pass(tmp_path):
    quota = run_eval.DailyQuotaExhausted("GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    records = run([quota, []], tmp_path)

    assert len(records) == 1 and records[0]["score"] is None


def test_tested_ragas_version_is_the_pinned_and_installed_one():
    requirements = Path(rejudge_faithfulness.__file__).with_name("requirements.txt")
    pin = f"ragas=={rejudge_faithfulness.TESTED_RAGAS}"

    assert pin in requirements.read_text(encoding="utf-8").splitlines()
    assert importlib.metadata.version("ragas") == rejudge_faithfulness.TESTED_RAGAS


def test_private_ragas_api_still_matches_what_the_script_calls():
    from ragas.metrics import Faithfulness
    from ragas.metrics._faithfulness import (
        NLIStatementOutput,
        StatementFaithfulnessAnswer,
        StatementGeneratorOutput,
    )

    def params(name):
        return list(inspect.signature(getattr(Faithfulness, name)).parameters)

    assert params("_create_statements") == ["self", "row", "callbacks"]
    assert params("_create_verdicts") == ["self", "row", "statements", "callbacks"]
    assert "statements" in StatementGeneratorOutput.model_fields
    verdicts = NLIStatementOutput(statements=[
        StatementFaithfulnessAnswer(statement="s1", reason="in context", verdict=1),
        StatementFaithfulnessAnswer(statement="s2", reason="not in context", verdict=0),
    ])
    assert Faithfulness()._compute_score(verdicts) == 0.5
    assert verdicts.statements[1].model_dump() == {"statement": "s2", "reason": "not in context", "verdict": 0}


def test_main_refuses_an_untested_ragas_version(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "9.9.9")
    monkeypatch.setattr(sys, "argv", ["rejudge", "results.jsonl", "--question", "q", "--out", str(out)])

    assert rejudge_faithfulness.main() == 2
    assert "9.9.9" in capsys.readouterr().out and not out.exists()

"""torch and sentence-transformers load only with EMBEDDING_PROVIDER=local.
With Gemini embeddings neither the backend (import plus startup) nor the
eval harness (which talks to a running backend, then scores with RAGAS)
imports them; loading torch costs about 0.5 GB of RAM per process.

Each check runs in a fresh interpreter: sys.modules is process-wide, so
inside pytest another test may already have imported torch."""
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]

HEAVY = ("torch", "sentence_transformers", "transformers")

REPORT = "import sys; print(sorted(m for m in {heavy!r} if m in sys.modules))"


def _loaded_modules(code: str, env: dict) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code + "\n" + REPORT.format(heavy=HEAVY)],
        cwd=BACKEND, env={**os.environ, **env}, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_backend_startup_with_gemini_embeddings_does_not_load_torch(pg_url):
    code = (
        "from fastapi.testclient import TestClient\n"
        "import deps, main\n"
        "with TestClient(main.app):\n"
        "    assert type(deps.get_embeddings()).__name__ == 'GeminiEmbeddings'\n"
    )
    env = {"DATABASE_URL": pg_url, "EMBEDDING_PROVIDER": "gemini", "EMBEDDING_MODEL": "",
           "EMBEDDING_DIMENSIONS": "0", "GEMINI_API_KEY": "test-key-not-real",
           "LLM_PROVIDER": "gemini", "PYTHONPATH": str(BACKEND)}

    assert _loaded_modules(code, env) == "[]"


def test_eval_harness_does_not_load_torch():
    # Everything run_eval imports on the way to scoring: the harness, the
    # Gemini judge and the RAGAS metrics. No call is made.
    code = (
        "import sys; sys.path.insert(0, 'eval')\n"
        "import run_eval\n"
        "run_eval.build_judge(); run_eval.judge_run_config()\n"
        "from ragas import EvaluationDataset, evaluate\n"
        "from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference\n"
    )
    env = {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key-not-real"}

    assert _loaded_modules(code, env) == "[]"

"""Replay frozen predictions against pinned sources, without model inference."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path


PINS = {
    "baseline": "d6de81643928d653435c431bae19945d41d32520",
    "fixed": "80b17f4a32b1aebb12302a2262962376e9d29d80",
}
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_digest(root):
    """Hash relative paths and bytes of every file in the archived package."""
    entries = [
        [path.relative_to(root).as_posix(), sha256(path.read_bytes())]
        for path in sorted((root / "lm_eval").rglob("*"))
        if path.is_file()
    ]
    return sha256(encoded(entries))


def shape(value):
    if isinstance(value, list):
        return [shape(item) for item in value]
    return type(value).__name__


def worker(source, destination):
    import sitecustomize

    if not sitecustomize.NETWORK_BLOCKED:
        raise RuntimeError("Network guard was not loaded")

    from datasets import Dataset, DatasetDict

    import lm_eval.api.task as task_module
    import lm_eval.evaluator_utils as aggregation_module
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.evaluator_utils import _compute_task_aggregations

    source = source.resolve()
    fixtures = json.loads((HERE / "fixtures.json").read_text())
    rows = []
    for case in fixtures["target_cases"]:
        for order in fixtures["metric_orders"]:
            calls = []

            def text_equal(references, predictions, _calls=calls):
                _calls.append(
                    {
                        "references": references,
                        "predictions": predictions,
                        "reference_shape": shape(references),
                        "prediction_shape": shape(predictions),
                    }
                )
                return float(predictions == references)

            dataset = DatasetDict(
                {
                    "test": Dataset.from_list(
                        [{"question": "Question", "target": case["target"]}]
                    )
                }
            )
            task = ConfigurableTask(
                config={
                    "task": "metric_order_replay",
                    "custom_dataset": lambda dataset=dataset, **kwargs: dataset,
                    "test_split": "test",
                    "doc_to_text": "question",
                    "doc_to_target": "target",
                    "output_type": "generate_until",
                    "metric_list": [
                        {
                            "metric": text_equal if name == "text_equal" else name,
                            "aggregation": "mean",
                            "higher_is_better": True,
                        }
                        for name in order
                    ],
                }
            )
            common = {
                "case": case["id"],
                "target": case["target"],
                "metric_order": order,
                "multiple_target": bool(task.multiple_target),
            }
            raw_metrics = defaultdict(list)
            for prediction in fixtures["predictions"]:
                calls.clear()
                scores = task.process_results(dataset["test"][0], [prediction])
                rows.append(
                    {
                        **common,
                        "kind": "document",
                        "prediction": prediction,
                        "scores": scores,
                        "text_equal_calls": list(calls),
                    }
                )
                for name, value in scores.items():
                    raw_metrics[(name, "none")].append(value)
            scores, samples = _compute_task_aggregations(
                task, raw_metrics, bootstrap_iters=0
            )
            rows.append(
                {**common, "kind": "aggregate", "samples": samples, "scores": scores}
            )

    # Verify every loaded package module belongs to this arm, including lazy imports.
    imports = {}
    for name, module in sorted(sys.modules.items()):
        if name == "lm_eval" or name.startswith("lm_eval."):
            path = getattr(module, "__file__", None)
            if path:
                resolved = Path(path).resolve()
                imports[name] = resolved.relative_to(source).as_posix()
    identity = {
        "imported_modules": imports,
        "source_sha256": source_digest(source),
        "task_sha256": sha256(Path(task_module.__file__).read_bytes()),
        "aggregation_sha256": sha256(Path(aggregation_module.__file__).read_bytes()),
    }
    destination.write_bytes(b"".join(encoded(row) for row in rows))
    destination.with_suffix(".identity.json").write_bytes(encoded(identity))


def validate(results):
    """Check observations against fixture truth and the specific baseline defect."""
    expectations = []
    for arm, rows in results.items():
        assert len(rows) == 64
        for row in rows:
            gold = row["target"]
            gold = gold if isinstance(gold, list) else [gold]
            corrupted = (
                arm == "baseline"
                and row["multiple_target"]
                and row["metric_order"] == ["exact_match", "text_equal"]
            )
            if row["kind"] == "document":
                truth = float(row["prediction"] in gold)
                for metric, score in row["scores"].items():
                    expected = 0.0 if corrupted and metric == "text_equal" else truth
                    assert score == expected, (arm, row, expected)
                for call in row["text_equal_calls"]:
                    expected_prediction = (
                        [[row["prediction"]] * len(gold)]
                        if corrupted
                        else [row["prediction"]]
                    )
                    assert call["predictions"] == expected_prediction, (arm, row)
            else:
                documents = [
                    item
                    for item in rows
                    if item["kind"] == "document"
                    and item["case"] == row["case"]
                    and item["metric_order"] == row["metric_order"]
                ]
                assert row["samples"] == len(documents) == 3
                for metric in row["metric_order"]:
                    expected = sum(item["scores"][metric] for item in documents) / 3
                    assert row["scores"][metric + ",none"] == expected
                    assert row["scores"][metric + "_stderr,none"] == "N/A"
                expectations.append({"arm": arm, **row})
    return expectations


def run(destination):
    if sys.version_info < (3, 12):
        raise RuntimeError("Use Python 3.12+; the recorded run uses 3.12.11")
    destination.mkdir(parents=True, exist_ok=False)
    results = {}
    runs = []
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to read the pinned source snapshots")
    with tempfile.TemporaryDirectory(prefix="metric-order-replay-") as temporary:
        scratch = Path(temporary)
        for arm, commit in PINS.items():
            resolved = subprocess.check_output(  # noqa: S603 -- fixed commit and no shell
                [git, "rev-parse", commit], cwd=REPO, text=True
            ).strip()
            assert resolved == commit
            archive = subprocess.check_output(  # noqa: S603 -- fixed commit and no shell
                [git, "archive", commit, "lm_eval"], cwd=REPO
            )
            source = scratch / arm
            source.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
                bundle.extractall(source, filter="data")
            source_hash = source_digest(source)
            for repeat in (1, 2):
                private = scratch / f"{arm}-{repeat}"
                private.mkdir()
                output = destination / f"{arm}-{repeat}.jsonl"
                env = {
                    "PATH": os.defpath,
                    "HOME": str(private),
                    "TMPDIR": str(private),
                    "LC_ALL": "C",
                    "PYTHONPATH": os.pathsep.join(
                        [str(HERE / "netguard"), str(source)]
                    ),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "HF_DATASETS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
                command = [
                    sys.executable,
                    str(HERE / "replay.py"),
                    "--worker",
                    str(source),
                    str(output),
                ]
                process = subprocess.run(  # noqa: S603 -- this script and no shell
                    command, cwd=private, env=env, capture_output=True, check=False
                )
                # Logs contain only package warnings; never record inherited environment values.
                (destination / f"{arm}-{repeat}.stdout.txt").write_bytes(process.stdout)
                (destination / f"{arm}-{repeat}.stderr.txt").write_bytes(process.stderr)
                if process.returncode:
                    raise RuntimeError(
                        f"{arm} repeat {repeat} failed; see captured stderr"
                    )
                identity = json.loads(output.with_suffix(".identity.json").read_text())
                assert identity["source_sha256"] == source_hash
                runs.append(
                    {
                        "arm": arm,
                        "repeat": repeat,
                        "commit": commit,
                        "exit_code": process.returncode,
                        "results_sha256": sha256(output.read_bytes()),
                        "source_sha256": source_hash,
                        "command_template": [
                            "<python>",
                            "<experiment>/replay.py",
                            "--worker",
                            "<archived-source>",
                            output.name,
                        ],
                    }
                )
            assert (destination / f"{arm}-1.jsonl").read_bytes() == (
                destination / f"{arm}-2.jsonl"
            ).read_bytes()
            assert (destination / f"{arm}-1.identity.json").read_bytes() == (
                destination / f"{arm}-2.identity.json"
            ).read_bytes()
            results[arm] = [
                json.loads(line)
                for line in (destination / f"{arm}-1.jsonl").read_text().splitlines()
            ]
    summary = validate(results)
    (destination / "aggregates.json").write_bytes(encoded(summary))
    environment = {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
            )
        ),
        "fixtures_sha256": sha256((HERE / "fixtures.json").read_bytes()),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "guard_sha256": sha256((HERE / "netguard/sitecustomize.py").read_bytes()),
        "runs": runs,
        "repeat_result_bytes_equal": True,
        "validation": "All document scores, callback inputs and aggregates match the specified expectations.",
    }
    (destination / "environment.json").write_bytes(encoded(environment))
    print(
        "Validated 16 configurations x 3 frozen predictions per arm; both fresh-process repeats match byte for byte."
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without -O: validation assertions must remain enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, type=Path, metavar=("SOURCE", "OUTPUT"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    elif args.output:
        run(args.output.resolve())
    else:
        parser.error("--output is required")

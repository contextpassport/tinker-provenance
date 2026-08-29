"""
Runs a LoRA fine-tune and records its provenance as a Context Passport chain.

    python -m src.run                      # offline, no API key, deterministic
    python -m src.run --tinker             # against the real Tinker API
    python -m src.run --out my.passports.json

Offline is the default on purpose. The provenance layer is the point of this
template, and it should be runnable, testable and reviewable by somebody who
has never had a Tinker key. The offline path produces the same chain shape as
a real run, with fixed values so that CI output is stable.

What this is not: a training recipe. The hyperparameters here are placeholders.
See the Tinker cookbook for how to actually train something worth deploying.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .provenance import (
    DATASET_PREPARED,
    DEPLOYMENT_APPROVED,
    EVAL_COMPLETED,
    FINETUNE_COMPLETED,
    FINETUNE_STARTED,
    SAFETY_RETESTED,
    ProvenanceChain,
    file_digest,
)

BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_RANK = 32
LEARNING_RATE = 1e-4
STEPS = 200


def _train_offline(dataset_path: Path) -> dict:
    """
    Simulate a run. Returns the same shape the Tinker path returns.

    Fixed numbers, not random ones: a template whose output changes every run
    cannot be diffed, and the committed example file would churn on every CI
    run for no reason.
    """
    return {
        "steps": STEPS,
        "final_loss": 0.412,
        "weights_id": "offline-simulated-weights",
        "backend": "offline-simulator",
    }


def _train_tinker(dataset_path: Path) -> dict:
    """
    The real thing, against the documented Tinker API.

    This is the only function in the template that talks to Tinker, which is
    deliberate: everything else works unchanged whether you train here, on your
    own hardware, or somewhere else entirely.

    Written against the published API surface (ServiceClient,
    create_lora_training_client, forward_backward, optim_step,
    save_weights_and_get_sampling_client). It has not been exercised against a
    live account by the authors of this template, so treat it as the shape to
    adapt rather than a recipe to trust blindly, and check it against the
    current docs at https://tinker-docs.thinkingmachines.ai/ before relying on
    the provenance it produces.
    """
    import tinker
    from tinker import types

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    examples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    final_loss = None
    for _ in range(STEPS):
        fwdbwd = training_client.forward_backward(examples, "cross_entropy")
        optim = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE))
        result = fwdbwd.result()
        optim.result()
        final_loss = getattr(result, "loss", None)

    sampling_client = training_client.save_weights_and_get_sampling_client("provenance-demo")

    return {
        "steps": STEPS,
        "final_loss": final_loss,
        "weights_id": getattr(sampling_client, "model_name", "provenance-demo"),
        "backend": "tinker",
    }


def _evaluate(weights_id: str) -> dict:
    """Placeholder evaluation. Replace with your actual benchmark harness."""
    return {
        "suite": "internal-qa-v3",
        "accuracy": 0.871,
        "baseline_accuracy": 0.844,
        "samples": 1200,
    }


def _safety_retest(weights_id: str) -> dict:
    """
    Re-run safety evaluations after fine-tuning.

    This record exists because fine-tuning can weaken a base model's safety
    behaviour, so a safety result measured before customisation does not carry
    over to the model you are about to deploy. Recording the retest is what
    turns "we checked" into something a reviewer can check back.
    """
    return {
        "suite": "safety-eval-v2",
        "passed": True,
        "refusal_rate": 0.962,
        "baseline_refusal_rate": 0.981,
        "regressions": ["refusal_rate down 1.9pp against base model"],
    }


def _dev_key():
    """
    A local Ed25519 key, generated on first use and reused after.

    This is a development key sitting unencrypted on disk, which is fine for
    demonstrating that signing works and unacceptable for anything you would
    show a regulator. In production the private key belongs in an HSM or a KMS,
    and the thing that makes a signature worth anything is that the operator
    cannot quietly re-sign a rewritten history.
    """
    from pathlib import Path

    from context_passport import generate_keypair
    from cryptography.hazmat.primitives import serialization

    key_dir = Path("./.keys")
    key_path = key_dir / "pipeline.key"
    key_dir.mkdir(exist_ok=True)

    if key_path.exists():
        private = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    else:
        private, _ = generate_keypair()
        key_path.write_bytes(private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        key_path.chmod(0o600)

    return private, "tinker-provenance-dev-key-1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune with a verifiable provenance chain.")
    parser.add_argument("--tinker", action="store_true", help="use the real Tinker API instead of the offline simulator")
    parser.add_argument("--dataset", default="examples/dataset.jsonl", help="training data (jsonl)")
    parser.add_argument("--out", default="run.passports.json", help="where to write the chain")
    parser.add_argument("--approver", default="ml-governance", help="who signs off on deployment")
    parser.add_argument("--sign", action="store_true", help="sign each record with a local Ed25519 dev key")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2

    if args.tinker and not os.environ.get("TINKER_API_KEY"):
        print("--tinker requires TINKER_API_KEY", file=sys.stderr)
        return 2

    signer, key_id = (_dev_key() if args.sign else (None, None))

    chain = ProvenanceChain(
        operator_id="tinker-provenance-template",
        operator_name="Fine-tuning Pipeline",
        trace_id="finetune-demo-001",
        base_model=BASE_MODEL,
        signer=signer,
        key_id=key_id,
    )

    # 1. What data went in. The digest, never the data itself.
    chain.record(DATASET_PREPARED, {
        "dataset": dataset.name,
        "digest": file_digest(dataset),
        "records": sum(1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()),
        "source": "internal support transcripts, redacted",
        "prepared_by": "data-platform",
    })

    # 2. What was asked of the trainer. Recorded before the run, so the
    #    configuration cannot be quietly rewritten to match a good result.
    chain.record(FINETUNE_STARTED, {
        "base_model": BASE_MODEL,
        "method": "lora",
        "rank": LORA_RANK,
        "learning_rate": LEARNING_RATE,
        "planned_steps": STEPS,
    })

    train = _train_tinker(dataset) if args.tinker else _train_offline(dataset)

    # 3. What came out.
    chain.record(FINETUNE_COMPLETED, train)

    # 4. Whether it is any good.
    evaluation = _evaluate(train["weights_id"])
    chain.record(EVAL_COMPLETED, evaluation)

    # 5. Whether it is still safe. Separate from evaluation on purpose: a model
    #    can get better at the task and worse at refusing, and one number
    #    covering both would hide exactly that.
    safety = _safety_retest(train["weights_id"])
    chain.record(SAFETY_RETESTED, safety)

    # 6. Who decided to ship it, knowing the above. If the safety retest had
    #    failed and a human shipped anyway, this record would use the
    #    "override" type instead, and the chain would carry that admission
    #    permanently. That is the point.
    chain.record(DEPLOYMENT_APPROVED, {
        "approved": True,
        "approver_ref": args.approver,
        "authority": "Model Risk Policy MRP-2026-04",
        "acknowledged_regressions": safety["regressions"],
        "weights_id": train["weights_id"],
    })

    out = chain.save(args.out)
    ok = chain.verify()

    print(f"  backend        {train['backend']}")
    print(f"  signed         {bool(signer)}")
    print(f"  records        {len(chain.passports)}")
    print(f"  chain intact   {ok}")
    print(f"  written        {out}")
    print()
    print(f"Verify it yourself:  python -m src.verify {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

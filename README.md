# Tinker Provenance

> A verifiable record of how a fine-tuned model came to exist: what data went in, what was run, whether it stayed safe, and who decided to ship it.

Fine-tuning a general-purpose model can make **you** the provider under the EU AI Act, not the lab whose weights you started from. Substantial modification moves the obligation. [Tinker](https://thinkingmachines.ai/tinker/) exists so you can fine-tune easily, which means Tinker's users acquire those obligations easily too.

The question an auditor asks is not "is your model good." It is:

> Prove this model came from that base model and that data, show me the safety results measured **after** you customised it, and show me who approved deploying it.

That is a chain of custody question. Today the answer usually lives in an experiment tracker, a few Slack threads, and somebody's memory. All of those are mutable and internal. You can edit an MLflow run. What you cannot quietly edit is a hash chain, which is what makes this worth handing to somebody who has no reason to trust you.

This template records a fine-tuning run as a chain of [Context Passport](https://contextpassport.com) records: an open CC0 standard, verifiable offline, with no server, no account, and no dependency on any vendor including the one that wrote this.

## Quick start

No API key required. The default run uses an offline simulator so the provenance layer is runnable by anyone.

```bash
pip install -r requirements.txt
python -m src.run
python -m src.verify run.passports.json
```

```
  backend        offline-simulator
  signed         False
  records        6
  chain intact   True
  written        run.passports.json
```

Against real Tinker:

```bash
export TINKER_API_KEY=...
python -m src.run --tinker --sign
```

## What gets recorded

Six records, each hashed to the one before it:

| # | Event type | What it pins down |
|---|---|---|
| 0 | `tinker.dataset_prepared` | sha256 of the training data, record count, who assembled it |
| 1 | `tinker.finetune_started` | base model, LoRA rank, learning rate, planned steps |
| 2 | `tinker.finetune_completed` | steps actually run, final loss, resulting weights id |
| 3 | `tinker.eval_completed` | benchmark scores against baseline |
| 4 | `tinker.safety_retested` | refusal rates **after** fine-tuning, and any regressions |
| 5 | `consent` | who approved deployment, under what authority, knowing the above |

Two of those are worth dwelling on.

**The dataset digest, not the dataset.** Training data is large, usually confidential, and often contains personal data that has no business being in an artifact designed to be shown to third parties. The digest proves which data was used to anyone still holding a copy and reveals nothing to anyone who is not.

**The safety retest is separate from the evaluation.** A fine-tune can make a model better at your task and worse at refusing, and a single quality number hides exactly that. Recording the retest is what turns "we checked" into something a reviewer can check back on.

Record 1 is written *before* training runs, so the configuration cannot be quietly rewritten afterwards to match a good result.

## Watch it catch a lie

```bash
python -m src.verify run.passports.json --tamper eval
```

```
  tampered with  tinker.eval_completed (accuracy raised to 0.999)
  chain intact   False

  Detected. The edited record and every record after it now fail.
```

That is the entire value proposition, and it takes four seconds to confirm rather than trust.

## Wiring it into your own pipeline

`src/provenance.py` knows nothing about Tinker. It records what a run reported, so it works unchanged whether you train on Tinker, on your own GPUs, or anywhere else. Provenance that only works with one vendor's API is not provenance, it is telemetry.

```python
from src.provenance import ProvenanceChain, FINETUNE_STARTED, file_digest

chain = ProvenanceChain(operator_id="ml-platform", operator_name="ML Platform")
chain.record(FINETUNE_STARTED, {"base_model": "...", "rank": 32})
# ... your training ...
chain.save("run.passports.json")
```

`src/run.py` is the part you replace. Only `_train_tinker()` touches Tinker at all.

### Recording an override

If the safety retest fails and a human ships anyway, that record should use the registered `override` event type rather than `consent`. The chain then carries the admission permanently, alongside who made the call and under what authority. That is not a flaw in the design. It is the reason to keep the record.

## Honest limits

- **The Tinker code path has not been run against a live account by the authors.** It is written against the published API surface: `ServiceClient`, `create_lora_training_client`, `forward_backward`, `optim_step`, `save_weights_and_get_sampling_client`. Check it against [the current docs](https://tinker-docs.thinkingmachines.ai/) before relying on the provenance it produces. The offline path is exercised in CI on every commit.
- **`--sign` uses a development key** sitting unencrypted in `./.keys/`. Fine for demonstrating that signing works; unacceptable for anything you would show a regulator. In production the key belongs in an HSM or KMS, and what makes a signature worth anything is that the operator cannot quietly re-sign a rewritten history.
- **This proves integrity, not completeness.** Nothing here can prove you recorded every run. It proves that the runs you did record have not been altered since. See [§5.4 of the specification](https://github.com/contextpassport/spec/blob/main/SPEC.md) for what to combine it with when completeness matters.
- **The hyperparameters are placeholders.** This is a provenance template, not a training recipe. See the [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) for the latter.

## Not affiliated with Thinking Machines Lab

This is an independent integration example built on a public open standard. Tinker and Inkling are products of Thinking Machines Lab, who have no involvement in this template and have not endorsed it.

## License

Apache-2.0. The Context Passport specification itself is CC0.

"""
Records a fine-tuning run as a Context Passport chain.

The recorder is deliberately ignorant of Tinker. It records what a run
reported, which means it works the same whether the training happened on
Tinker, on your own GPUs, or in the offline simulator in run.py. Provenance
that only works with one vendor's API is not provenance, it is telemetry.

Each record hashes the one before it. Change any record after the fact and
every record following it stops verifying, which is the property that makes
the chain worth handing to somebody who has no reason to trust you.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from context_passport import make_passport, verify_chain

# Event types are namespaced per SPEC.md 3.3, which permits custom types and
# asks that they be prefixed. The namespace is not decoration: it is what makes
# these records findable. A generic "commit" is indistinguishable from every
# other passport in the world, whereas "tinker.finetune_started" appears only
# where somebody actually recorded a fine-tuning run.
NS = "tinker"

DATASET_PREPARED = f"{NS}.dataset_prepared"
FINETUNE_STARTED = f"{NS}.finetune_started"
FINETUNE_COMPLETED = f"{NS}.finetune_completed"
EVAL_COMPLETED = f"{NS}.eval_completed"
SAFETY_RETESTED = f"{NS}.safety_retested"

# Approval uses the registered compliance type from SPEC.md 3.3 rather than a
# namespaced one. A human signing off on a deployment is not Tinker-specific,
# and a verifier that understands the specification should recognise it without
# knowing anything about this template.
DEPLOYMENT_APPROVED = "consent"
DEPLOYMENT_OVERRIDE = "override"


def file_digest(path: str | Path) -> str:
    """
    sha256 of a file, prefixed the way the specification writes hashes.

    The training data itself does not go in the record. It is usually large,
    frequently confidential, and often contains personal data that has no
    business being in an artifact designed to be shown to third parties. The
    digest proves which data was used to anyone who still holds a copy, and
    reveals nothing to anyone who does not.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


class ProvenanceChain:
    """
    Builds and holds an ordered chain of passports for one training run.

    Usage:
        chain = ProvenanceChain(operator_id="ml-platform", operator_name="ML Platform")
        chain.record(DATASET_PREPARED, {...})
        chain.record(FINETUNE_STARTED, {...})
        chain.save("run.passports.json")
    """

    def __init__(
        self,
        *,
        operator_id: str,
        operator_name: str,
        trace_id: Optional[str] = None,
        base_model: Optional[str] = None,
        signer: Optional[Any] = None,
        key_id: Optional[str] = None,
    ) -> None:
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.trace_id = trace_id
        self.base_model = base_model
        self.signer = signer
        self.key_id = key_id
        self.passports: list[dict] = []

    def record(self, event_type: str, payload: dict) -> dict:
        """Append one record, chained to the previous one."""
        parent = self.passports[-1] if self.passports else None
        passport = make_passport(
            self.operator_id,
            self.operator_name,
            payload,
            parent=parent,
            event_type=event_type,
            trace_id=self.trace_id,
            provider="thinking-machines",
            model=self.base_model,
        )

        if self.signer is not None:
            from context_passport import sign_passport

            passport = sign_passport(passport, self.signer, key_id=self.key_id)

        self.passports.append(passport)
        return passport

    def verify(self) -> bool:
        """Recompute every hash and confirm each record links to its parent."""
        return verify_chain(self.passports)

    def save(self, path: str | Path) -> Path:
        """
        Write the chain as a single file.

        The .passports.json suffix is the plural form recommended by SPEC.md
        3.5, for a file holding a chain rather than one record. It carries no
        weight in verification and exists so that editor tooling recognises the
        file without being configured to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.passports, f, indent=2, sort_keys=True)
            f.write("\n")
        return path

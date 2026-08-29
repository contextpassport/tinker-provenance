"""
Verifies a fine-tuning provenance chain offline.

    python -m src.verify run.passports.json
    python -m src.verify run.passports.json --tamper eval

No network, no account, no dependency on whoever produced the file. That is
the whole claim: a reviewer who does not trust you can still check this.

--tamper edits one record in memory and verifies again, so you can watch the
chain break rather than take anyone's word that it would.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from context_passport import verify_chain


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path} does not contain a chain (expected a JSON array)")
    return data


def summarise(passports: list[dict]) -> None:
    print(f"  {len(passports)} records")
    for i, p in enumerate(passports):
        event = p.get("event", {}).get("type", "?")
        payload = p.get("payload", {})
        # One useful detail per event type, chosen to be the thing a reviewer
        # would actually look for rather than the first key in the dict.
        detail = (
            payload.get("digest")
            or payload.get("base_model")
            or payload.get("approver_ref")
            or payload.get("suite")
            or payload.get("weights_id")
            or ""
        )
        print(f"   {i}. {event:<28} {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a provenance chain.")
    parser.add_argument("path", nargs="?", default="run.passports.json")
    parser.add_argument(
        "--tamper",
        metavar="SUBSTRING",
        help="edit the first record whose event type contains SUBSTRING, then verify again",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}. Run `python -m src.run` first.", file=sys.stderr)
        return 2

    passports = load(path)
    summarise(passports)

    ok = verify_chain(passports)
    print()
    print(f"  chain intact   {ok}")

    if not args.tamper:
        return 0 if ok else 1

    edited = copy.deepcopy(passports)
    target = next(
        (p for p in edited if args.tamper in p.get("event", {}).get("type", "")),
        None,
    )
    if target is None:
        print(f"no record matching {args.tamper!r}", file=sys.stderr)
        return 2

    # Change something a person would plausibly want to change after the fact:
    # a number that makes the run look better than it was.
    payload = target.setdefault("payload", {})
    if "accuracy" in payload:
        payload["accuracy"] = 0.999
        what = "accuracy raised to 0.999"
    elif "passed" in payload:
        payload["passed"] = True
        what = "safety marked as passed"
    else:
        payload["_edited"] = True
        what = "payload edited"

    still_ok = verify_chain(edited)
    print()
    print(f"  tampered with  {target['event']['type']} ({what})")
    print(f"  chain intact   {still_ok}")

    if still_ok:
        print()
        print("  PROBLEM: tampering was not detected. Do not rely on this chain.", file=sys.stderr)
        return 1

    print()
    print("  Detected. The edited record and every record after it now fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Interactively create the private API-key file without echoing or overwriting it."""
from __future__ import annotations

import argparse
import getpass
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="WO_AGENT_API_KEY",
                        help="Target environment variable name in .secrets/scenecode.env (default: WO_AGENT_API_KEY)")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite an existing .secrets/scenecode.env atomically with mode 0600")
    parser.add_argument("--from-env", dest="from_env", default=None,
                        help="Read the API key non-interactively from the given host env var (e.g. OPENAI_API_KEY)")
    args = parser.parse_args()
    if not ENV_NAME.fullmatch(args.name):
        parser.error("--name must be a valid environment-variable name")
    directory = ROOT / ".secrets"
    target = directory / "scenecode.env"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    if (target.exists() or target.is_symlink()) and not args.replace:
        parser.error(".secrets/scenecode.env already exists; pass --replace to update it")
    if args.from_env:
        if not ENV_NAME.fullmatch(args.from_env):
            parser.error("--from-env must be a valid environment-variable name")
        value = os.environ.get(args.from_env, "").strip()
        if not value:
            parser.error(f"environment variable {args.from_env} is empty or unset")
    else:
        value = getpass.getpass("GPT / OpenAI-compatible Agent API key (input hidden): ").strip()
    if not value or "\n" in value or "\r" in value or "\x00" in value:
        parser.error("API key must be a nonempty single-line value")
    if args.replace:
        fd, tmp_name = tempfile.mkstemp(prefix=".scenecode.env.", dir=directory, text=True)
        try:
            os.chmod(tmp_name, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(f"{args.name}={value}\n")
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        print("Updated .secrets/scenecode.env with mode 0600. The value was not echoed.")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(target, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(f"{args.name}={value}\n")
        except Exception:
            # Preserve any partially written file for manual inspection; never unlink it here.
            raise
        print("Created .secrets/scenecode.env with mode 0600. The value was not echoed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

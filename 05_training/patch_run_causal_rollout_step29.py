"""
Patch run_causal_rollout.py with a safe A-family bridge dispatch.

The patch inserts a small top-level dispatcher after the module docstring
and after any __future__ imports.

It only activates when --a-family-bridge is present in sys.argv.
Otherwise, the original run_causal_rollout.py behavior is unchanged.
"""

from __future__ import annotations

from pathlib import Path


TARGET = Path("05_training/run_causal_rollout.py")
MARKER_START = "# [STEP29_A_FAMILY_BRIDGE_DISPATCH_START]"
MARKER_END = "# [STEP29_A_FAMILY_BRIDGE_DISPATCH_END]"


DISPATCH_BLOCK = '''
# [STEP29_A_FAMILY_BRIDGE_DISPATCH_START]
def _urbanbus_step29_maybe_dispatch_a_family_bridge():
    """
    Dispatch to Step 28 A-family bridge entrypoint only when
    --a-family-bridge is explicitly provided.

    This keeps the original run_causal_rollout.py path unchanged.
    """
    import sys
    from pathlib import Path

    if "--a-family-bridge" not in sys.argv:
        return

    sys.argv = [arg for arg in sys.argv if arg != "--a-family-bridge"]

    training_dir = Path(__file__).resolve().parent
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))

    from run_causal_rollout_a_family_bridge_v1 import main as _a_family_bridge_main

    raise SystemExit(_a_family_bridge_main())


_urbanbus_step29_maybe_dispatch_a_family_bridge()
# [STEP29_A_FAMILY_BRIDGE_DISPATCH_END]

'''


def find_docstring_end(lines: list[str], start: int) -> int:
    """
    If a module docstring starts at lines[start], return index after it.
    Otherwise return start.
    """
    if start >= len(lines):
        return start

    stripped = lines[start].lstrip()

    if not (stripped.startswith('"""') or stripped.startswith("'''")):
        return start

    quote = '"""' if stripped.startswith('"""') else "'''"

    # One-line docstring.
    remainder = stripped[len(quote):]
    if quote in remainder:
        return start + 1

    i = start + 1
    while i < len(lines):
        if quote in lines[i]:
            return i + 1
        i += 1

    raise RuntimeError("unterminated module docstring")


def find_insert_index(lines: list[str]) -> int:
    i = 0

    # Shebang.
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1

    # Encoding comments and leading comments/blanks before docstring.
    while i < len(lines):
        stripped = lines[i].strip()
        lower = stripped.lower()
        if stripped == "":
            i += 1
            continue
        if lower.startswith("# -*- coding") or lower.startswith("# coding"):
            i += 1
            continue
        break

    # Module docstring.
    i = find_docstring_end(lines, i)

    # Blanks after docstring.
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # __future__ imports must stay before normal code.
    while i < len(lines) and lines[i].startswith("from __future__ import"):
        i += 1

    # Include blank lines after future imports.
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    return i


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"[FAIL] target not found: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if MARKER_START in text:
        print("[OK] Step 29 dispatch already present; no patch needed")
        return

    lines = text.splitlines(keepends=True)
    idx = find_insert_index(lines)

    new_lines = lines[:idx] + [DISPATCH_BLOCK] + lines[idx:]
    new_text = "".join(new_lines)

    TARGET.write_text(new_text, encoding="utf-8", newline="")

    print("[OK] patched:", TARGET)
    print("[OK] insert_index:", idx)


if __name__ == "__main__":
    main()

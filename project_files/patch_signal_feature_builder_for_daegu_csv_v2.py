from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER = PROJECT_ROOT / "05_training" / "adapters" / "build_signal_features_for_causal_simulator_v2.py"


def replace_block(text: str, name: str, replacement: str) -> str:
    pattern = rf"{name}\s*=\s*\[.*?\]\n"
    new_text, n = re.subn(pattern, replacement + "\n", text, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(f"failed to replace block: {name}")
    return new_text


def patch() -> None:
    if not BUILDER.exists():
        raise RuntimeError(f"builder not found: {BUILDER}")

    text = BUILDER.read_text(encoding="utf-8-sig")

    text = replace_block(
        text,
        "ID_CANDIDATES",
        'ID_CANDIDATES = [\n'
        '    "signal_id", "id", "신호등id", "신호등_id", "신호등관리번호", "시설물관리번호", "관리번호"\n'
        ']',
    )
    text = replace_block(
        text,
        "TYPE_CANDIDATES",
        'TYPE_CANDIDATES = [\n'
        '    "signal_type", "type", "신호등종류", "신호등구분", "종류", "시설구분", "신호기종류", "신호제어방식"\n'
        ']',
    )
    text = replace_block(
        text,
        "NAME_CANDIDATES",
        'NAME_CANDIDATES = [\n'
        '    "signal_name", "name", "신호등명", "시설명", "교차로명", "설치위치", "소재지도로명주소", "소재지지번주소", "도로노선명"\n'
        ']',
    )

    new_func = r"""def _yes_no_flag_from_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col].astype(str).str.strip().str.upper()
    return s.isin(["Y", "YES", "TRUE", "1", "O", "○", "있음", "유", "운영", "작동"])


def _nonempty_metadata_flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col].astype(str).str.strip()
    return s.notna() & (s != "") & (s.str.lower() != "nan")


def add_signal_flags(df: pd.DataFrame, inferred: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    type_col = inferred.get("type_col")
    name_col = inferred.get("name_col")

    text = pd.Series([""] * len(out), index=out.index, dtype="object")
    if type_col and type_col in out.columns:
        text = text + " " + out[type_col].astype(str)
    if name_col and name_col in out.columns:
        text = text + " " + out[name_col].astype(str)

    for extra_col in [
        "신호등구분",
        "신호등색종류",
        "신호등화방식",
        "신호제어방식",
        "신호시간결정방식",
        "소재지도로명주소",
        "소재지지번주소",
    ]:
        if extra_col in out.columns:
            text = text + " " + out[extra_col].astype(str)

    lowered = text.str.lower()

    pedestrian_from_text = (
        lowered.str.contains("보행", regex=False)
        | lowered.str.contains("pedestrian", regex=False)
        | lowered.str.contains("횡단", regex=False)
    )
    pedestrian_from_cols = (
        _yes_no_flag_from_col(out, "보행자작동신호기유무")
        | _yes_no_flag_from_col(out, "시각장애인용음향신호기유무")
    )

    blink_from_text = (
        lowered.str.contains("점멸", regex=False)
        | lowered.str.contains("blink", regex=False)
        | lowered.str.contains("flashing", regex=False)
    )
    blink_from_cols = _yes_no_flag_from_col(out, "점멸등운영여부")

    controlled_from_text = (
        lowered.str.contains("제어", regex=False)
        | lowered.str.contains("control", regex=False)
        | lowered.str.contains("controlled", regex=False)
        | lowered.str.contains("신호", regex=False)
    )
    controlled_from_cols = (
        _nonempty_metadata_flag(out, "신호제어방식")
        | _nonempty_metadata_flag(out, "신호시간결정방식")
    )

    out["_pedestrian_signal_flag"] = (pedestrian_from_text | pedestrian_from_cols).astype(int)
    out["_blink_signal_flag"] = (blink_from_text | blink_from_cols).astype(int)
    out["_controlled_signal_flag"] = (controlled_from_text | controlled_from_cols).astype(int)

    return out


"""

    pattern = r"def add_signal_flags\(df: pd\.DataFrame, inferred: Dict\[str, Optional\[str\]\]\) -> pd\.DataFrame:\n.*?\n\ndef prepare_signal_xy"
    replacement = new_func + "\ndef prepare_signal_xy"
    text, n = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError("failed to replace add_signal_flags function")

    marker = "daegu_csv_step99_patch_v2"
    if marker not in text:
        text = text.replace(
            'FORBIDDEN_DYNAMIC_SIGNAL_FIELDS = [',
            '# daegu_csv_step99_patch_v2: Korean Daegu traffic signal CSV columns supported.\nFORBIDDEN_DYNAMIC_SIGNAL_FIELDS = [',
            1,
        )

    BUILDER.write_text(text, encoding="utf-8")
    print("[OK] patched Step 98 signal feature builder for Daegu signal CSV columns")
    print(f"[OK] builder: {BUILDER}")


if __name__ == "__main__":
    patch()

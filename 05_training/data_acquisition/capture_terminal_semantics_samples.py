from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_BASE_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
ARTIFACT_VERSION = "terminal_semantics_position_capture_v1"
KEY_VALUE_RE = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY|serviceKey)\s*[:=]\s*([A-Za-z0-9%+/=_\\-]+)")
POSITION_COLUMNS = [
    "request_time_kst",
    "provider_event_time",
    "route_id",
    "direction_id",
    "vehicle_id",
    "current_sequence",
    "current_stop_id",
    "x",
    "y",
    "response_status",
    "raw_file_path",
    "raw_sha256",
    "batch_id",
    "sampling_interval_seconds",
]


def kst_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def parse_routes(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            clean = part.strip()
            if clean and clean not in out:
                out.append(clean)
    return out


def read_text_maybe_rtf(path: Path) -> str:
    if path.suffix.lower() == ".rtf":
        result = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)], check=True, capture_output=True, text=True)
        return result.stdout
    return path.read_text(encoding="utf-8", errors="ignore")


def read_service_key(path: Optional[str], explicit: str) -> Tuple[str, str]:
    if explicit:
        return explicit, "argument"
    if path:
        text = read_text_maybe_rtf(Path(path).expanduser())
        match = KEY_VALUE_RE.search(text)
        if match:
            return match.group(1).strip(), "file"
        candidates = re.findall(r"[A-Za-z0-9%+/=_\\-]{20,}", text)
        if candidates:
            return max(candidates, key=len).strip(), "file"
    env_key = os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY") or ""
    return env_key, "environment" if env_key else "missing"


def redact_url(url: str, service_key: str) -> str:
    encoded = urllib.parse.quote(service_key, safe="%")
    encoded_plus = urllib.parse.quote_plus(service_key, safe="%")
    return url.replace(service_key, "<REDACTED>").replace(encoded, "<REDACTED>").replace(encoded_plus, "<REDACTED>")


def detect_status(raw_text: str, http_status: Optional[int]) -> str:
    stripped = (raw_text or "").strip()
    low = stripped.lower()
    if http_status == 429:
        return "RATE_LIMIT"
    if not stripped:
        return "EMPTY_RESPONSE"
    if stripped.startswith("<!doctype html") or stripped.startswith("<html") or "<html" in low[:500]:
        return "HTML_RESPONSE"
    if any(marker in low for marker in ["servicekey", "service key", "인증키", "등록되지 않은", "unauthorized", "not authorized", "invalid"]):
        return "AUTH_ERROR"
    if any(marker in low for marker in ["service error", "application error", "internal server error", "시스템 오류", "서비스 오류"]):
        return "PROVIDER_ERROR"
    if http_status is None:
        return "NETWORK_ERROR"
    if http_status >= 400:
        return "HTTP_ERROR"
    return "OK"


def find_items(obj: Any) -> List[Dict[str, Any]]:
    paths = [
        ["body", "items"],
        ["body", "items", "item"],
        ["response", "body", "items"],
        ["response", "body", "items", "item"],
        ["items"],
        ["item"],
    ]
    for path in paths:
        cur = obj
        for part in path:
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = None
                break
        if isinstance(cur, list):
            return [row for row in cur if isinstance(row, dict)]
        if isinstance(cur, dict):
            return [cur]
    rows: List[Dict[str, Any]] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if any(k in value for k in ["routeId", "bsId", "vhcNo", "vhcNo2", "xPos", "yPos", "seq"]):
                rows.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(obj)
    return rows


def parse_rows(raw_text: str) -> List[Dict[str, Any]]:
    stripped = raw_text.lstrip()
    try:
        if stripped.startswith("{") or stripped.startswith("["):
            return find_items(json.loads(raw_text))
        if stripped.startswith("<"):
            root = ET.fromstring(raw_text)
            rows = []
            for item in root.findall(".//item"):
                row = {child.tag.split("}")[-1]: child.text for child in list(item)}
                if row:
                    rows.append(row)
            return rows
    except Exception:
        return []
    return []


def norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_rows(route_id: str, raw_rows: Sequence[Mapping[str, Any]], raw_path: Path, raw_sha256: str, request_time: str, status: str, batch_id: str, interval: int) -> List[Dict[str, Any]]:
    rows = []
    for row in raw_rows:
        rows.append(
            {
                "request_time_kst": request_time,
                "provider_event_time": norm(row.get("arTime")) or norm(row.get("eventTime")) or norm(row.get("tm")),
                "route_id": norm(row.get("routeId")) or route_id,
                "direction_id": norm(row.get("moveDir")) or norm(row.get("direction_id")),
                "vehicle_id": norm(row.get("vhcNo2")) or norm(row.get("vhcNo")) or norm(row.get("busId")),
                "current_sequence": norm(row.get("seq")) or norm(row.get("stopSeq")),
                "current_stop_id": norm(row.get("bsId")) or norm(row.get("stopId")),
                "x": norm(row.get("xPos")) or norm(row.get("x")) or norm(row.get("longitude")),
                "y": norm(row.get("yPos")) or norm(row.get("y")) or norm(row.get("latitude")),
                "response_status": status,
                "raw_file_path": str(raw_path),
                "raw_sha256": raw_sha256,
                "batch_id": batch_id,
                "sampling_interval_seconds": interval,
            }
        )
    return rows


def fetch_route(base_url: str, service_key: str, route_id: str, output_dir: Path, batch_id: str, interval: int, timeout: int = 30) -> Dict[str, Any]:
    request_time = kst_now()
    raw_dir = output_dir / "raw" / batch_id / route_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = [("serviceKey", service_key), ("routeId", route_id), ("resultType", "json")]
    url = base_url + "?" + urllib.parse.urlencode(params, safe="%")
    raw_path = raw_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    http_status: Optional[int] = None
    content_type = ""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-terminal-semantics/1.0"}), timeout=timeout) as response:
            http_status = int(getattr(response, "status", 200) or 200)
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
            encoding = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        http_status = int(exc.code)
        payload = exc.read() or b""
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        encoding = exc.headers.get_content_charset() if exc.headers else "utf-8"
    except Exception as exc:
        payload = json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False).encode("utf-8")
        content_type = "application/json"
        encoding = "utf-8"
    raw_path.write_bytes(payload)
    raw_text = payload.decode(encoding or "utf-8", errors="replace")
    status = detect_status(raw_text, http_status)
    raw_sha = sha256_file(raw_path)
    rows = normalize_rows(route_id, parse_rows(raw_text), raw_path, raw_sha, request_time, status, batch_id, interval)
    return {
        "route_id": route_id,
        "request_time_kst": request_time,
        "request_url_redacted": redact_url(url, service_key),
        "http_status": http_status,
        "content_type": content_type,
        "response_status": status,
        "raw_file_path": str(raw_path),
        "raw_sha256": raw_sha,
        "normalized_row_count": len(rows),
        "normalized_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture terminal operation semantics live position samples.")
    parser.add_argument("--route-ids", nargs="+", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-calls-per-minute", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--service-key", default="")
    parser.add_argument("--service-key-file", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    route_ids = parse_routes(args.route_ids)
    service_key, key_source = read_service_key(args.service_key_file, args.service_key)
    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_kst": kst_now(),
        "route_ids": route_ids,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "interval_seconds": args.interval_seconds,
        "batch_id": args.batch_id,
        "max_calls_per_minute": args.max_calls_per_minute,
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "service_key_present": bool(service_key),
        "service_key_source": key_source,
        "service_key_literal_recorded": False,
        "rate_limit_guard_enabled": True,
        "requests": [],
    }
    rows: List[Dict[str, Any]] = []
    if args.dry_run or not service_key:
        manifest["status"] = "DRY_RUN" if args.dry_run else "MISSING_SERVICE_KEY"
    else:
        start_time = parse_time(args.start_time)
        end_time = parse_time(args.end_time)
        min_call_sleep = max(0.0, 60.0 / max(args.max_calls_per_minute, 1))
        sample_index = 0
        while datetime.now().astimezone() < end_time:
            now = datetime.now().astimezone()
            if now < start_time:
                time.sleep(min(30.0, max(0.1, (start_time - now).total_seconds())))
                continue
            sample_index += 1
            sample_started = time.monotonic()
            for route_id in route_ids:
                result = fetch_route(args.base_url, service_key, route_id, output_dir, args.batch_id, args.interval_seconds)
                request = {k: v for k, v in result.items() if k != "normalized_rows"}
                request["sample_index"] = sample_index
                manifest["requests"].append(request)
                rows.extend(result["normalized_rows"])
                sleep_for = max(min_call_sleep * 2.0, 10.0) if result["response_status"] == "RATE_LIMIT" else min_call_sleep
                time.sleep(min(sleep_for, max(0.0, (end_time - datetime.now().astimezone()).total_seconds())))
                if datetime.now().astimezone() >= end_time:
                    break
            remaining = args.interval_seconds - (time.monotonic() - sample_started)
            if remaining > 0:
                time.sleep(min(remaining, max(0.0, (end_time - datetime.now().astimezone()).total_seconds())))
        manifest["status"] = "PASS" if rows else "NO_NORMALIZED_ROWS"
    pd.DataFrame(rows, columns=POSITION_COLUMNS).to_parquet(output_dir / "terminal_semantics_position_samples.parquet", index=False)
    dump_json(output_dir / "capture_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "normalized_rows": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

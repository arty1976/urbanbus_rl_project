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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


DEFAULT_BASE_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
ARTIFACT_VERSION = "terminal_position_capture_v1"
POSITION_SAMPLE_COLUMNS = [
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
]
KEY_VALUE_RE = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY|serviceKey)\s*[:=]\s*([A-Za-z0-9%+/=_\\-]+)")


def kst_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def read_text_maybe_rtf(path: Path) -> str:
    if path.suffix.lower() == ".rtf":
        try:
            result = subprocess.run(
                ["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except Exception:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            raw = re.sub(r"\\'[0-9a-fA-F]{2}", "", raw)
            raw = re.sub(r"\\[a-zA-Z]+-?[0-9]* ?", "", raw)
            return re.sub(r"[{}]", " ", raw)
    return path.read_text(encoding="utf-8", errors="ignore")


def read_service_key_file(path: Optional[str]) -> str:
    if not path:
        return ""
    text = read_text_maybe_rtf(Path(path).expanduser())
    match = KEY_VALUE_RE.search(text)
    if match:
        return match.group(1).strip()
    candidates = re.findall(r"[A-Za-z0-9%+/=_\\-]{20,}", text)
    return max(candidates, key=len).strip() if candidates else ""


def normalize_json_rows(route_id: str, raw_text: str, raw_file_path: Path, raw_sha256: str, request_time_kst: str, response_status: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(raw_text)
    except Exception:
        obj = None
    rows: List[Dict[str, Any]] = []
    candidates: List[Any] = []
    if isinstance(obj, dict):
        for path in (["body", "items"], ["response", "body", "items"], ["body", "items", "item"], ["response", "body", "items", "item"], ["items"]):
            cur: Any = obj
            for key in path:
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = None
                    break
            if isinstance(cur, list):
                candidates = cur
                break
            if isinstance(cur, dict):
                candidates = [cur]
                break
    for item in candidates:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "request_time_kst": request_time_kst,
                "provider_event_time": item.get("arTime") or item.get("eventTime") or item.get("tm"),
                "route_id": str(item.get("routeId") or item.get("route_id") or route_id),
                "direction_id": item.get("moveDir") or item.get("direction_id"),
                "vehicle_id": item.get("vhcNo") or item.get("vhcNo2") or item.get("vehicle_id") or item.get("busId"),
                "current_sequence": item.get("seq") or item.get("stopSeq") or item.get("route_sequence"),
                "current_stop_id": item.get("bsId") or item.get("stopId") or item.get("stop_id"),
                "x": item.get("xPos") or item.get("x") or item.get("longitude"),
                "y": item.get("yPos") or item.get("y") or item.get("latitude"),
                "response_status": response_status,
                "raw_file_path": str(raw_file_path),
                "raw_sha256": raw_sha256,
            }
        )
    return rows


def normalize_xml_rows(route_id: str, raw_text: str, raw_file_path: Path, raw_sha256: str, request_time_kst: str, response_status: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(raw_text)
    except Exception:
        return []
    rows = []
    for item in root.findall(".//item"):
        row = {child.tag: child.text for child in list(item)}
        rows.extend(normalize_json_rows(route_id, json.dumps({"items": [row]}, ensure_ascii=False), raw_file_path, raw_sha256, request_time_kst, response_status))
    return rows


def normalize_rows(route_id: str, raw_text: str, raw_file_path: Path, raw_sha256: str, request_time_kst: str, response_status: str) -> List[Dict[str, Any]]:
    stripped = raw_text.lstrip()
    if stripped.startswith("<"):
        return normalize_xml_rows(route_id, raw_text, raw_file_path, raw_sha256, request_time_kst, response_status)
    return normalize_json_rows(route_id, raw_text, raw_file_path, raw_sha256, request_time_kst, response_status)


def fetch_route(base_url: str, service_key: str, route_id: str, output_dir: Path, timeout: int = 20) -> Dict[str, Any]:
    request_time = kst_now()
    raw_dir = output_dir / "raw" / route_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = {"serviceKey": service_key, "routeId": route_id, "resultType": "json"}
    url = base_url + "?" + urllib.parse.urlencode(params, safe="%")
    raw_path = raw_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-terminal-capture/1.0"}), timeout=timeout) as response:
            payload = response.read()
            status = int(getattr(response, "status", 200) or 200)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        payload = exc.read() or b""
        status = int(exc.code)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
    except Exception as exc:
        payload = json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False).encode("utf-8")
        status = -1
        content_type = "application/json"
    raw_path.write_bytes(payload)
    text = payload.decode("utf-8", errors="replace")
    provider_status = "RATE_LIMIT" if status == 429 else ("AUTH_OR_PROVIDER_ERROR" if "SERVICE_KEY" in text.upper() or "인증" in text else ("OK" if 200 <= status < 300 else "HTTP_ERROR"))
    rows = normalize_rows(route_id, text, raw_path, sha256_file(raw_path), request_time, provider_status)
    return {
        "route_id": route_id,
        "request_time_kst": request_time,
        "http_status": status,
        "content_type": content_type,
        "raw_file_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "response_status": provider_status,
        "normalized_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Daegu BIS terminal position samples.")
    parser.add_argument("--route-ids", nargs="+", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-calls-per-minute", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--service-key", default="")
    parser.add_argument("--service-key-file", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    route_ids = parse_routes(args.route_ids)
    service_key = args.service_key or read_service_key_file(args.service_key_file) or os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY") or ""
    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_kst": kst_now(),
        "route_ids": route_ids,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "interval_seconds": args.interval_seconds,
        "max_calls_per_minute": args.max_calls_per_minute,
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "base_url": args.base_url,
        "service_key_present": bool(service_key),
        "service_key_source": "file" if args.service_key_file and service_key else ("argument" if args.service_key else ("environment" if service_key else "missing")),
        "rate_limit_guard_enabled": True,
        "status": "DRY_RUN" if args.dry_run else "PENDING",
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
                result = fetch_route(args.base_url, service_key, route_id, output_dir)
                request = {k: v for k, v in result.items() if k != "normalized_rows"}
                request["sample_index"] = sample_index
                request["normalized_row_count"] = len(result["normalized_rows"])
                manifest["requests"].append(request)
                rows.extend(result["normalized_rows"])
                if result["http_status"] == 429:
                    time.sleep(max(min_call_sleep * 2.0, 10.0))
                else:
                    time.sleep(min_call_sleep)
            elapsed = time.monotonic() - sample_started
            remaining = max(0.0, args.interval_seconds - elapsed)
            if remaining:
                time.sleep(min(remaining, max(0.0, (end_time - datetime.now().astimezone()).total_seconds())))
        manifest["status"] = "PASS" if rows else "NO_NORMALIZED_ROWS"
    pd.DataFrame(rows, columns=POSITION_SAMPLE_COLUMNS).to_parquet(output_dir / "normalized_position_samples.parquet", index=False)
    dump_json(output_dir / "capture_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "normalized_rows": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Batch runner: runs a folder (or one file) of test inputs against the API and writes everything to one folder.

Inputs it understands:
  *.mp3 / *.wav / *.m4a   uploaded as voiceovers
  *.txt                   sent as a script
  *.json                  a full request body (BrollRequest); may contain "file": "relative/audio.mp3" to upload
  *.jsonl                 one request body per line

Usage:
  python -m cli.batch_runner testpack/inputs --api http://localhost:8000 --out testpack/outputs [--token T] [--concurrency 2]
"""
import argparse
import csv
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

AUDIO = {".mp3", ".wav", ".m4a"}


def collect(src: Path) -> list[tuple[str, dict, Path | None]]:
    files = sorted(src.iterdir()) if src.is_dir() else [src]
    cases = []
    for f in files:
        ext = f.suffix.lower()
        if ext in AUDIO:
            cases.append((f.stem, {}, f))
        elif ext == ".txt":
            cases.append((f.stem, {"script": f.read_text(encoding="utf-8")}, None))
        elif ext == ".json":
            body = json.loads(f.read_text(encoding="utf-8"))
            audio = body.pop("file", None)
            cases.append((f.stem, body, (f.parent / audio) if audio else None))
        elif ext == ".jsonl":
            for i, line in enumerate(l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()):
                body = json.loads(line)
                audio = body.pop("file", None)
                cases.append((f"{f.stem}_{i:02d}", body, (f.parent / audio) if audio else None))
        elif not f.is_dir():
            # anything else is sent as an upload so wrong-file-type handling is tested too
            cases.append((f.stem + ext.replace(".", "_"), {}, f))
    return cases


def run_case(api: str, headers: dict, out: Path, name: str, body: dict, audio: Path | None, timeout: int) -> dict:
    case_dir = out / name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "request.json").write_text(json.dumps({**body, "file": str(audio) if audio else None}, indent=2))
    h = {**headers, "Idempotency-Key": f"batch-{name}-{uuid.uuid4().hex[:8]}", "X-Correlation-Id": f"batch-{name}"}
    t0 = time.time()
    row = {"case": name, "status": "", "error_code": "", "seconds": 0.0, "scenes": "", "narration_seconds": "",
           "qa_passed": "", "warnings": 0, "job_id": ""}
    with httpx.Client(timeout=120) as c:
        if audio:
            r = c.post(f"{api}/api/v1/broll/jobs", headers=h,
                       files={"file": (audio.name, audio.read_bytes())}, data={"request": json.dumps(body)})
        else:
            r = c.post(f"{api}/api/v1/broll/jobs", headers=h, json=body)
        if r.status_code != 202:
            row.update(status="rejected", error_code=r.json().get("error", {}).get("code", r.status_code))
            (case_dir / "response.json").write_text(json.dumps(r.json(), indent=2))
            row["seconds"] = round(time.time() - t0, 2)
            return row
        job_id = r.json()["job_id"]
        row["job_id"] = job_id
        while True:
            try:
                j = c.get(f"{api}/api/v1/jobs/{job_id}", headers=headers).json()
            except httpx.TransportError:  # server restarting: keep polling (jobs are requeued on restart)
                time.sleep(5)
                if time.time() - t0 > timeout:
                    j = {"status": "timeout"}
                    break
                continue
            if j["status"] in ("succeeded", "failed") or time.time() - t0 > timeout:
                break
            time.sleep(2)
        row["seconds"] = round(time.time() - t0, 2)
        (case_dir / "job.json").write_text(json.dumps(j, indent=2))
        row["status"] = j["status"]
        if j["status"] == "succeeded":
            res = j["result"]
            row.update(scenes=res.get("scenes"), narration_seconds=res.get("narration_seconds"),
                       qa_passed=res.get("qa_passed"), warnings=len(res.get("warnings", [])))
            for key, link in res["links"].items():
                url = link if link.startswith("http") else api + link
                data = c.get(url, headers=headers).content
                fname = Path(url.split("?")[0]).name
                (case_dir / fname).write_bytes(data)
        elif j.get("error"):
            row["error_code"] = j["error"]["code"]
        else:
            row["status"] = "timeout"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--out", type=Path, default=Path("batch_results"))
    ap.add_argument("--token", default="")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {a.token}"} if a.token else {}
    cases = collect(a.inputs)
    print(f"{len(cases)} cases -> {a.out}")
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(run_case, a.api.rstrip("/"), headers, a.out, n, b, f, a.timeout) for n, b, f in cases]
        rows = []
        for fut in futs:
            try:
                row = fut.result()
            except Exception as e:
                row = {"case": "?", "status": "runner_error", "error_code": type(e).__name__}
            rows.append(row)
            print(f"  {row.get('case'):30s} {row.get('status'):10s} {row.get('error_code') or ''} {row.get('seconds', '')}s")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=2))
    ok = sum(r.get("status") == "succeeded" for r in rows)
    print(f"done: {ok}/{len(rows)} succeeded; summary at {a.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

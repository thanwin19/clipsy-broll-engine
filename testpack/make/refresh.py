"""Re-downloads timeline.json and manifest.json for every finished case in batch output folders.

The independent VLM judge runs after a video is delivered and then updates timeline.json/manifest.json in
place (same links), so copies saved by the batch runner at delivery time do not have the judge scores yet.
Usage:  python -m testpack.make.refresh testpack/outputs_run4 [--api http://127.0.0.1:8000]"""
import json
import sys
from pathlib import Path

import httpx


def main(args):
    api = args[args.index("--api") + 1] if "--api" in args else None
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--api")]
    for d in dirs:
        for case in sorted(Path(d).iterdir()):
            j = case / "job.json"
            if not j.exists():
                continue
            job = json.loads(j.read_text())
            links = (job.get("result") or {}).get("links") or {}
            for key, name in (("timeline", "timeline.json"), ("manifest", "manifest.json")):
                url = links.get(key)
                if not url:
                    continue
                if api and "/files/" in url:
                    url = api.rstrip("/") + url[url.index("/files/"):]
                r = httpx.get(url, timeout=60)
                r.raise_for_status()
                (case / name).write_bytes(r.content)
            judged = json.loads((case / "manifest.json").read_text()).get("independent_judge") if links else None
            print(case.name, "judge:", judged)


if __name__ == "__main__":
    main(sys.argv[1:])

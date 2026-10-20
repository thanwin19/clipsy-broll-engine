"""Computes the Metrics-tab numbers from batch outputs:  python -m testpack.make.metrics testpack/outputs_run1 testpack/outputs_run2
Writes testpack/METRICS.md and testpack/metrics.json. Uses only recorded files (no estimates)."""
import json
import sys
from pathlib import Path


def load(dirs):
    cases = {}
    for d in dirs:
        for case in sorted(Path(d).iterdir()):
            if not case.is_dir():
                continue
            j = case / "job.json"
            r = case / "response.json"
            entry = {"run": Path(d).name, "case": case.name}
            if j.exists():
                job = json.loads(j.read_text())
                entry.update(status=job["status"], error=(job.get("error") or {}).get("code"))
                if job["status"] == "succeeded":
                    entry["timeline"] = json.loads((case / "timeline.json").read_text())
                    entry["manifest"] = json.loads((case / "manifest.json").read_text())
            elif r.exists():
                entry.update(status="rejected", error=json.loads(r.read_text())["error"]["code"])
            cases.setdefault(case.name, []).append(entry)
    return cases


def main(dirs):
    cases = load(dirs)
    rows, scene_scores, ok, total_attempts, repeats, scenes_total = [], [], 0, 0, 0, 0
    covered_by_clip, narration_total = 0.0, 0.0
    for name, attempts in cases.items():
        for a in attempts:
            total_attempts += 1
            if a.get("status") == "succeeded":
                ok += 1
                tl, m = a["timeline"], a["manifest"]
                ids = [s["clip"]["id"] for s in tl["scenes"]]
                repeats += len(ids) - len(set(ids))
                scenes_total += len(ids)
                narration_total += tl["narration_seconds"]
                for s in tl["scenes"]:
                    if s["relevance_score"] is not None:
                        scene_scores.append(s["relevance_score"])
                    covered_by_clip += s["duration"]
                rows.append({"case": name, "run": a["run"], "narration_s": tl["narration_seconds"],
                             "scenes": len(ids), "render_total_s": m["timings_seconds"].get("total"),
                             "seconds_per_minute": round(m["timings_seconds"].get("total", 0) / max(tl["narration_seconds"], 1) * 60, 1),
                             "qa_passed": all(r["passed"] for r in m["qa"].values()),
                             "sources": sorted({s["clip"]["source"] for s in tl["scenes"]}),
                             "ai_scenes": sum(s["clip"]["ai_generated"] for s in tl["scenes"]),
                             "cards": sum(s["clip"]["kind"] == "card" for s in tl["scenes"]),
                             "warnings": len(m["warnings"])})
            else:
                rows.append({"case": name, "run": a["run"], "status": a.get("status"), "error": a.get("error")})
    produced = [r for r in rows if "narration_s" in r]
    out = {
        "attempts": total_attempts, "succeeded": ok,
        "videos_produced": len(produced),
        "repeat_clip_rate": round(repeats / scenes_total, 4) if scenes_total else None,
        "scenes": scenes_total,
        "qa_pass_rate": round(sum(r["qa_passed"] for r in produced) / len(produced), 3) if produced else None,
        "mean_clip_score_stock_ai": round(sum(scene_scores) / len(scene_scores), 4) if scene_scores else None,
        "median_seconds_per_minute_of_video": sorted(r["seconds_per_minute"] for r in produced)[len(produced) // 2] if produced else None,
        "narration_seconds_total": round(narration_total, 1),
        "rows": rows,
    }
    Path("testpack/metrics.json").write_text(json.dumps(out, indent=2))
    lines = ["# Measured metrics (from recorded outputs)", "",
             f"- Attempts: {out['attempts']}, succeeded: {out['succeeded']} (expected rejections counted as not succeeded)",
             f"- Videos produced: {out['videos_produced']}, scenes: {out['scenes']}, repeat-clip rate: {out['repeat_clip_rate']}",
             f"- QA pass rate (black/frozen frames, drift, codec, fps): {out['qa_pass_rate']}",
             f"- Mean CLIP relevance of stock/AI scenes: {out['mean_clip_score_stock_ai']}",
             f"- Median processing seconds per minute of narration: {out['median_seconds_per_minute_of_video']}", "",
             "| case | run | narration s | scenes | total s | s per min | QA | sources | cards | warnings / error |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "narration_s" in r:
            lines.append(f"| {r['case']} | {r['run']} | {r['narration_s']} | {r['scenes']} | {r['render_total_s']} | "
                         f"{r['seconds_per_minute']} | {'pass' if r['qa_passed'] else 'FAIL'} | {', '.join(r['sources'])} | "
                         f"{r['cards']} | {r['warnings']} |")
        else:
            lines.append(f"| {r['case']} | {r['run']} | – | – | – | – | – | – | – | {r['status']}: {r['error']} |")
    Path("testpack/METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main(sys.argv[1:])

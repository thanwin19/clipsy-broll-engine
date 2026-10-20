"""Computes the Metrics-tab numbers from batch outputs:  python -m testpack.make.metrics testpack/outputs_run1 testpack/outputs_run2
Writes testpack/METRICS.md and testpack/metrics.json. Uses only recorded files (no estimates).

Human ratings: `--template` writes testpack/human_ratings.csv with one row per produced scene (run, case,
scene, start, end, narration, clip source, VLM judge score) and an empty `rating` column. Fill it with your own
1-5 ratings after watching the videos; the next run adds coverage, the share of scenes rated 4-5 and the
agreement between the VLM judge and the human ratings to METRICS.md."""
import csv
import json
import sys
from pathlib import Path

RATINGS = Path("testpack/human_ratings.csv")


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


template = False


def main(dirs):
    cases = load(dirs)
    rows, scene_scores, ok, total_attempts, repeats, scenes_total = [], [], 0, 0, 0, 0
    scene_rows = []
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
                    judge = (s.get("judge") or {}).get("score") if isinstance(s.get("judge"), dict) else s.get("judge")
                    scene_rows.append({"run": a["run"], "case": name, "scene": s["scene"], "start": s["start"],
                                       "end": s["end"], "narration": s["narration"], "source": s["clip"]["source"],
                                       "kind": s["clip"]["kind"], "judge": judge})
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
    judged = [r["judge"] for r in scene_rows if isinstance(r["judge"], (int, float))]
    out["vlm_judge"] = {"scenes_judged": len(judged),
                        "mean": round(sum(judged) / len(judged), 2) if judged else None,
                        "share_4_or_5": round(sum(j >= 4 for j in judged) / len(judged), 3) if judged else None}
    jt = [(r["end"] - r["start"], r["judge"]) for r in scene_rows if isinstance(r["judge"], (int, float))]
    out["judge_coverage_3_plus"] = round(sum(d for d, j in jt if j >= 3) / sum(d for d, _ in jt), 3) if jt else None
    out["human"] = human_metrics(scene_rows)
    if template:
        write_template(scene_rows)
    Path("testpack/metrics.json").write_text(json.dumps(out, indent=2))
    lines = ["# Measured metrics (from recorded outputs)", "",
             f"- Attempts: {out['attempts']}, succeeded: {out['succeeded']} (expected rejections counted as not succeeded)",
             f"- Videos produced: {out['videos_produced']}, scenes: {out['scenes']}, repeat-clip rate: {out['repeat_clip_rate']}",
             f"- QA pass rate (black/frozen frames, drift, codec, fps): {out['qa_pass_rate']}",
             f"- Mean CLIP relevance of stock/AI scenes: {out['mean_clip_score_stock_ai']}",
             f"- Median processing seconds per minute of narration: {out['median_seconds_per_minute_of_video']}",
             f"- Independent VLM judge (Gemini, 1-5): {out['vlm_judge']['scenes_judged']} scenes, mean "
             f"{out['vlm_judge']['mean']}, share rated 4-5: {out['vlm_judge']['share_4_or_5']}",
             human_line(out["human"]), "",
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


def write_template(scene_rows):
    old = {}
    if RATINGS.exists():   # keep ratings already entered
        for r in csv.DictReader(RATINGS.open(encoding="utf-8")):
            old[(r["run"], r["case"], str(r["scene"]))] = r.get("rating", "")
    with RATINGS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "case", "scene", "start", "end", "narration", "source", "kind",
                                          "judge", "rating"])
        w.writeheader()
        for r in scene_rows:
            w.writerow({**r, "rating": old.get((r["run"], r["case"], str(r["scene"])), "")})
    print(f"wrote {RATINGS} ({len(scene_rows)} scenes); fill the rating column with 1-5")


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return round(cov / (va * vb), 3) if va and vb else None


def human_metrics(scene_rows):
    if not RATINGS.exists():
        return None
    rated = {}
    for r in csv.DictReader(RATINGS.open(encoding="utf-8")):
        if (r.get("rating") or "").strip():
            rated[(r["run"], r["case"], str(r["scene"]))] = float(r["rating"])
    rows = [(r, rated[(r["run"], r["case"], str(r["scene"]))]) for r in scene_rows
            if (r["run"], r["case"], str(r["scene"])) in rated]
    if not rows:
        return None
    secs = sum(r["end"] - r["start"] for r, _ in rows)
    pairs = [(h, r["judge"]) for r, h in rows if isinstance(r["judge"], (int, float))]
    return {"scenes_rated": len(rows),
            "mean": round(sum(h for _, h in rows) / len(rows), 2),
            "share_4_or_5": round(sum(h >= 4 for _, h in rows) / len(rows), 3),
            "coverage_rated_3_plus": round(sum(r["end"] - r["start"] for r, h in rows if h >= 3) / secs, 3) if secs else None,
            "judge_pairs": len(pairs),
            "judge_exact_agreement": round(sum(h == j for h, j in pairs) / len(pairs), 3) if pairs else None,
            "judge_within_1": round(sum(abs(h - j) <= 1 for h, j in pairs) / len(pairs), 3) if pairs else None,
            "judge_spearman": spearman([h for h, _ in pairs], [j for _, j in pairs]) if pairs else None}


def human_line(h):
    if not h:
        return "- Human ratings: not entered yet (run with --template, fill testpack/human_ratings.csv)"
    return (f"- Human ratings (author, 1-5): {h['scenes_rated']} scenes, mean {h['mean']}, share rated 4-5: "
            f"{h['share_4_or_5']}, coverage by clips rated 3+: {h['coverage_rated_3_plus']}; VLM judge vs human on "
            f"{h['judge_pairs']} scenes: exact {h['judge_exact_agreement']}, within 1 point {h['judge_within_1']}, "
            f"Spearman {h['judge_spearman']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    template = "--template" in args
    main([a for a in args if a != "--template"])

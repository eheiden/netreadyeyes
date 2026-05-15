import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None

from netrunner_scanner.config import CARD_IMAGE_DIRS, CARD_IMAGE_EXTENSIONS
from netrunner_scanner.gpu_status import configure_gpu, get_gpu_status
from netrunner_scanner.catalog import CardCatalog, normalize_card_id


def candidate_filenames(card_id):
    card_id = str(card_id)
    names = [card_id]

    # Common card id format: sure_gamble_30030
    if "_" in card_id:
        names.append(card_id.rsplit("_", 1)[0])

    return list(dict.fromkeys(names))


def find_image_for_card_id(card_id, image_dirs):
    for folder in image_dirs:
        folder = Path(folder)
        if not folder.exists():
            continue

        for stem in candidate_filenames(card_id):
            for ext in CARD_IMAGE_EXTENSIONS:
                exact = folder / f"{stem}{ext}"
                if exact.exists():
                    return exact

                exact_lower = folder / f"{stem}{ext.lower()}"
                if exact_lower.exists():
                    return exact_lower

        # Fallback: recursive exact-stem search.
        wanted = set(candidate_filenames(card_id))
        for path in folder.rglob("*"):
            if path.suffix.lower() not in {ext.lower() for ext in CARD_IMAGE_EXTENSIONS}:
                continue
            if path.stem in wanted:
                return path

    return None


def load_catalog_ids(catalog_path):
    data = np.load(catalog_path, allow_pickle=True)
    return [str(card_id) for card_id in data["ids"]]


def build_cardpool_cases(catalog_path, image_dirs, limit):
    ids = load_catalog_ids(catalog_path)
    cases = []
    missing = []

    seen_paths = set()
    for card_id in ids:
        path = find_image_for_card_id(card_id, image_dirs)
        if path is None:
            missing.append(card_id)
            continue

        key = str(path.resolve()).lower()
        if key in seen_paths:
            continue

        seen_paths.add(key)
        cases.append({"expected_id": card_id, "path": str(path)})

        if limit and len(cases) >= limit:
            break

    return cases, missing


def same_card_name(a, b):
    return normalize_card_id(a) == normalize_card_id(b)


def filtered_different_name_results(results, expected_id):
    expected_name = normalize_card_id(expected_id)
    filtered = []

    for result in results:
        result_id = str(result.get("id"))
        if normalize_card_id(result_id) == expected_name:
            continue
        filtered.append(result)

    return filtered


def top_result_summary(results, expected_id):
    expected_id = str(expected_id)

    raw_top = results[0] if results else {"id": "none", "score": 0.0}
    raw_second = results[1] if len(results) > 1 else {"id": "none", "score": 0.0}

    expected_rank = None
    expected_score = None
    expected_name_rank = None
    expected_name_score = None

    for idx, result in enumerate(results, start=1):
        result_id = str(result.get("id"))
        if result_id == expected_id and expected_rank is None:
            expected_rank = idx
            expected_score = float(result.get("score", 0.0))
        if same_card_name(result_id, expected_id) and expected_name_rank is None:
            expected_name_rank = idx
            expected_name_score = float(result.get("score", 0.0))

    different_name_results = filtered_different_name_results(results, expected_id)
    top = different_name_results[0] if different_name_results else {"id": "none", "score": 0.0}
    second = different_name_results[1] if len(different_name_results) > 1 else {"id": "none", "score": 0.0}

    expected_name_score_for_margin = expected_name_score if expected_name_score is not None else 0.0
    top_score = float(top.get("score", 0.0))
    second_score = float(second.get("score", 0.0))
    margin_vs_expected_name = expected_name_score_for_margin - top_score

    correct_name_top = expected_name_rank == 1
    exact_printing_top = expected_rank == 1

    return {
        "expected_id": expected_id,
        "raw_top_id": str(raw_top.get("id")),
        "raw_top_score": float(raw_top.get("score", 0.0)),
        "raw_second_id": str(raw_second.get("id")),
        "raw_second_score": float(raw_second.get("score", 0.0)),
        "top_different_name_id": str(top.get("id")),
        "top_different_name_score": top_score,
        "second_different_name_id": str(second.get("id")),
        "second_different_name_score": second_score,
        "margin": margin_vs_expected_name,
        "expected_rank": expected_rank,
        "expected_score": expected_score,
        "expected_name_rank": expected_name_rank,
        "expected_name_score": expected_name_score,
        "correct_top": correct_name_top,
        "exact_printing_top": exact_printing_top,
        "same_name_printing_conflict": same_card_name(raw_top.get("id"), expected_id) and str(raw_top.get("id")) != expected_id,
        "ambiguity_score": margin_vs_expected_name,
    }



def get_process_cpu_times():
    if psutil is None:
        return None

    try:
        process = psutil.Process(os.getpid())
        times = process.cpu_times()
        return float(times.user + times.system)
    except Exception:
        return None


def cpu_count():
    return max(os.cpu_count() or 1, 1)


def summarize_cpu(start_cpu, end_cpu, wall_seconds):
    if start_cpu is None or end_cpu is None or wall_seconds <= 0:
        return {
            "cpu_supported": False,
            "cpu_time_seconds": None,
            "cpu_core_equivalent_percent": None,
            "cpu_whole_machine_percent": None,
            "logical_cpus": cpu_count(),
        }

    cpu_seconds = max(0.0, end_cpu - start_cpu)
    core_equiv = (cpu_seconds / wall_seconds) * 100.0
    whole_machine = core_equiv / cpu_count()

    return {
        "cpu_supported": True,
        "cpu_time_seconds": cpu_seconds,
        "cpu_core_equivalent_percent": core_equiv,
        "cpu_whole_machine_percent": whole_machine,
        "logical_cpus": cpu_count(),
    }

def run_single_mode(args):
    configure_gpu(args.single_mode == "on")
    provider_status = get_gpu_status()

    cases_path = Path(args.cases_json)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    catalog = CardCatalog(args.catalog)

    warmup = cases[: min(3, len(cases))]
    for case in warmup:
        img = Image.open(case["path"]).convert("RGB")
        catalog.search_image(img)

    times = []
    details = []

    wall_start = time.perf_counter()
    cpu_start = get_process_cpu_times()

    for case in cases:
        img = Image.open(case["path"]).convert("RGB")
        start = time.perf_counter()
        results = catalog.search_image(img)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

        summary = top_result_summary(results, case["expected_id"])
        summary["path"] = case["path"]
        summary["elapsed_ms"] = elapsed
        details.append(summary)

    cpu_end = get_process_cpu_times()
    wall_seconds = time.perf_counter() - wall_start
    cpu_summary = summarize_cpu(cpu_start, cpu_end, wall_seconds)

    output = {
        "mode": args.single_mode,
        "provider_status": provider_status,
        "image_count": len(cases),
        "mean_ms": statistics.mean(times) if times else 0.0,
        "median_ms": statistics.median(times) if times else 0.0,
        "min_ms": min(times) if times else 0.0,
        "max_ms": max(times) if times else 0.0,
        "wall_seconds": wall_seconds,
        "cpu": cpu_summary,
        "details": details,
    }

    Path(args.json_out).write_text(json.dumps(output, indent=2), encoding="utf-8")


def format_cpu_summary(result):
    cpu = result.get("cpu") or {}

    if not cpu.get("cpu_supported"):
        return "CPU=n/a (install psutil)"

    return (
        f"CPU={cpu.get('cpu_core_equivalent_percent', 0.0):.1f}% core-equiv "
        f"({cpu.get('cpu_whole_machine_percent', 0.0):.1f}% of machine, "
        f"{cpu.get('cpu_time_seconds', 0.0):.2f}s CPU)"
    )


def format_mode_summary(result):
    return (
        f"{result['mode'].upper()} | "
        f"provider={result['provider_status'].get('active')} | "
        f"mean={result['mean_ms']:.2f} ms | "
        f"median={result['median_ms']:.2f} ms | "
        f"min={result['min_ms']:.2f} | max={result['max_ms']:.2f} | "
        f"{format_cpu_summary(result)}"
    )


def difficulty_rows(details, limit, include_same_name_printings=False):
    rows = list(details)

    if not include_same_name_printings:
        # Same-name printing collisions are expected and not useful for practical gameplay.
        rows = [
            row for row in rows
            if not row.get("same_name_printing_conflict", False)
        ]

    # Sort by wrong card-name matches first, then smallest margin vs closest different-name card.
    return sorted(
        rows,
        key=lambda row: (
            0 if not row.get("correct_top") else 1,
            row.get("margin", 999.0),
            row.get("top_different_name_score", 0.0),
        ),
    )[:limit]


def write_report(report_path, cases, missing, results, difficult_limit, include_same_name_printings=False):
    lines = []
    lines.append("Net Ready Eyes benchmark report")
    lines.append("=" * 36)
    lines.append(f"Images tested: {len(cases)}")
    lines.append(f"Missing catalogue images: {len(missing)}")
    lines.append("")

    lines.append("Timing")
    lines.append("-" * 36)
    for result in results:
        lines.append(format_mode_summary(result))
    lines.append("")

    if len(results) == 2:
        on = next((r for r in results if r["mode"] == "on"), None)
        off = next((r for r in results if r["mode"] == "off"), None)
        if on and off and on["mean_ms"] > 0:
            delta = off["mean_ms"] - on["mean_ms"]
            pct = (delta / off["mean_ms"]) * 100.0 if off["mean_ms"] else 0.0
            if delta > 0:
                lines.append(f"GPU ON was faster by {delta:.2f} ms/image ({pct:.1f}%).")
            elif delta < 0:
                lines.append(f"GPU ON was slower by {abs(delta):.2f} ms/image ({abs(pct):.1f}%).")
            else:
                lines.append("GPU ON and OFF had the same mean timing.")

            on_cpu = (on.get("cpu") or {})
            off_cpu = (off.get("cpu") or {})
            if on_cpu.get("cpu_supported") and off_cpu.get("cpu_supported"):
                cpu_delta = off_cpu.get("cpu_core_equivalent_percent", 0.0) - on_cpu.get("cpu_core_equivalent_percent", 0.0)
                machine_delta = off_cpu.get("cpu_whole_machine_percent", 0.0) - on_cpu.get("cpu_whole_machine_percent", 0.0)

                if cpu_delta > 0:
                    lines.append(
                        f"GPU ON used less CPU by {cpu_delta:.1f} core-equivalent percentage points "
                        f"({machine_delta:.1f} whole-machine points)."
                    )
                elif cpu_delta < 0:
                    lines.append(
                        f"GPU ON used more CPU by {abs(cpu_delta):.1f} core-equivalent percentage points "
                        f"({abs(machine_delta):.1f} whole-machine points)."
                    )
                else:
                    lines.append("GPU ON and OFF used the same measured CPU load.")
            else:
                lines.append("CPU-load comparison unavailable; install psutil to enable it.")

            lines.append("")

    # Use GPU-on details if available; otherwise first mode.
    base = next((r for r in results if r["mode"] == "on"), results[0])
    hard = difficulty_rows(base["details"], difficult_limit, include_same_name_printings=include_same_name_printings)

    same_name_conflicts = sum(1 for row in base["details"] if row.get("same_name_printing_conflict"))
    lines.append("Most difficult / ambiguous cards")
    lines.append("-" * 36)
    if include_same_name_printings:
        lines.append("Same-name printing conflicts are included in this list.")
    else:
        lines.append("Same-name printing conflicts are excluded from this list.")
    lines.append(f"Excluded same-name printing conflicts: {same_name_conflicts}")
    lines.append("")

    for idx, row in enumerate(hard, start=1):
        name_rank = row["expected_name_rank"] if row["expected_name_rank"] is not None else "not in top results"
        exact_rank = row["expected_rank"] if row["expected_rank"] is not None else "not in top results"
        lines.append(
            f"{idx:02d}. expected={row['expected_id']} | "
            f"closest_different_name={row['top_different_name_id']} ({row['top_different_name_score']:.3f}) | "
            f"second_different_name={row['second_different_name_id']} ({row['second_different_name_score']:.3f}) | "
            f"margin_vs_expected_name={row['margin']:.3f} | "
            f"expected_name_rank={name_rank} | exact_printing_rank={exact_rank} | "
            f"path={row['path']}"
        )

    if missing:
        lines.append("")
        lines.append("Missing image examples")
        lines.append("-" * 36)
        for card_id in missing[:50]:
            lines.append(card_id)
        if len(missing) > 50:
            lines.append(f"...and {len(missing) - 50} more")

    text = "\n".join(lines)
    Path(report_path).write_text(text, encoding="utf-8")
    return text


def run_parent(args):
    image_dirs = args.image_dirs or CARD_IMAGE_DIRS
    cases, missing = build_cardpool_cases(args.catalog, image_dirs, args.limit)

    if not cases:
        raise SystemExit(
            "No benchmark images found. Make sure your card image folders exist. "
            f"Searched: {', '.join(map(str, image_dirs))}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cases_json = tmp / "cases.json"
        cases_json.write_text(json.dumps(cases, indent=2), encoding="utf-8")

        results = []
        for mode in ("on", "off"):
            json_out = tmp / f"{mode}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-mode",
                mode,
                "--catalog",
                args.catalog,
                "--cases-json",
                str(cases_json),
                "--json-out",
                str(json_out),
            ]
            print(f"\nRunning GPU {mode.upper()} benchmark...")
            subprocess.run(cmd, check=True)
            result = json.loads(json_out.read_text(encoding="utf-8"))
            results.append(result)
            print(format_mode_summary(result))

    report_text = write_report(
        args.report,
        cases,
        missing,
        results,
        args.difficult_limit,
        include_same_name_printings=args.include_same_name_printings,
    )

    print("")
    print(report_text)
    print("")
    print(f"Wrote report: {args.report}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Net Ready Eyes GPU/CPU and card ambiguity.")
    parser.add_argument("--catalog", default="netrunner-catalog.npz", help="Catalogue npz path.")
    parser.add_argument(
        "--image-dirs",
        nargs="*",
        default=None,
        help="Image folders to search. Defaults to CARD_IMAGE_DIRS from config.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cards. 0 = all found cardpool images.")
    parser.add_argument("--report", default="benchmark_report.txt", help="Output report text file.")
    parser.add_argument("--difficult-limit", type=int, default=25, help="How many difficult cards to list.")
    parser.add_argument(
        "--include-same-name-printings",
        action="store_true",
        help="Include cases where the top result is a different printing of the same card name.",
    )

    # Internal subprocess args.
    parser.add_argument("--single-mode", choices=["on", "off"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cases-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json-out", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.single_mode:
        run_single_mode(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()

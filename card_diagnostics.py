
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from netrunner_scanner.catalog import CardCatalog, normalize_card_id
from netrunner_scanner.config import CARD_IMAGE_DIRS, CARD_IMAGE_EXTENSIONS
from netrunner_scanner.recognition import recognize_candidate_crop


def find_card_image(card_id, image_dirs):
    wanted = str(card_id)
    wanted_name = normalize_card_id(wanted)

    for folder in image_dirs:
        folder = Path(folder)
        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if path.suffix.lower() not in {ext.lower() for ext in CARD_IMAGE_EXTENSIONS}:
                continue
            if path.stem == wanted:
                return path
            if normalize_card_id(path.stem) == wanted_name:
                return path

    return None


def pil_to_frame(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def candidate_for_image(pil_image):
    w, h = pil_image.size
    box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.intp)
    return {
        "box": box,
        "area": float(w * h),
        "source": "diagnostic_static",
    }


def make_variants(pil_image):
    base = pil_image.convert("RGB")
    variants = [("original", base)]

    for angle in (-20, -10, 10, 20):
        variants.append((f"rotate_{angle}", base.rotate(angle, expand=True, fillcolor=(32, 32, 32))))

    variants.append(("darker", ImageEnhance.Brightness(base).enhance(0.72)))
    variants.append(("brighter", ImageEnhance.Brightness(base).enhance(1.25)))
    variants.append(("lower_contrast", ImageEnhance.Contrast(base).enhance(0.75)))
    variants.append(("blur_1px", base.filter(ImageFilter.GaussianBlur(radius=1.0))))

    return variants


def summarize_result(result):
    if result is None:
        return {
            "id": None,
            "score": 0.0,
            "margin": None,
            "reason": "no_result",
            "alternatives": [],
        }

    return {
        "id": result.get("id"),
        "score": float(result.get("score", 0.0)),
        "margin": result.get("margin"),
        "reason": result.get("refine_reason"),
        "used_fallback": result.get("used_fallback"),
        "sharpness": result.get("sharpness"),
        "alternatives": result.get("alternatives") or [],
    }


def run(args):
    image_dirs = args.image_dirs or CARD_IMAGE_DIRS
    card_path = Path(args.image) if args.image else find_card_image(args.card, image_dirs)

    if card_path is None or not card_path.exists():
        raise SystemExit(f"Could not find image for {args.card!r}. Try --image path/to/card.jpg")

    catalog = CardCatalog(args.catalog)
    pil_image = Image.open(card_path).convert("RGB")

    out_dir = Path(args.out) / normalize_card_id(args.card or card_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for name, variant in make_variants(pil_image):
        frame = pil_to_frame(variant)
        candidate = candidate_for_image(variant)

        result = recognize_candidate_crop(
            frame=frame,
            candidate=candidate,
            side="diagnostic",
            candidate_index=0,
            catalog=catalog,
            force_diagnostics=True,
        )

        variant.save(out_dir / f"{name}.jpg")

        summary = summarize_result(result)
        summary["variant"] = name
        rows.append(summary)

    report = {
        "card": args.card,
        "image": str(card_path),
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"Card diagnostics for {args.card}")
    lines.append(f"Image: {card_path}")
    lines.append("")
    for row in rows:
        margin = row["margin"]
        margin_text = "n/a" if margin is None else f"{float(margin):.3f}"
        lines.append(
            f"{row['variant']:16} -> {row['id']} "
            f"score={row['score']:.3f} margin={margin_text} reason={row['reason']}"
        )
        for alt in row["alternatives"][:5]:
            lines.append(f"    {alt['id']} {float(alt['score']):.3f} rot={alt.get('rotation')}")
    text = "\n".join(lines)

    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print("")
    print(f"Wrote diagnostics to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run targeted diagnostics for a problem card.")
    parser.add_argument("--card", required=True, help="Card id/name, e.g. cezve_33017 or cezve")
    parser.add_argument("--image", default=None, help="Optional explicit image path.")
    parser.add_argument("--catalog", default="netrunner-catalog.npz")
    parser.add_argument("--out", default="diagnostics")
    parser.add_argument("--image-dirs", nargs="*", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

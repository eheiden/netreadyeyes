import argparse
from netrunner_scanner.build_catalog import build_catalog
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or update a CollectorVision Netrunner catalog.")
    parser.add_argument("--source", choices=["folder", "nrdb"], default="nrdb")
    parser.add_argument("--image-dir", default="cards")
    parser.add_argument("--alt-art-dir", default="alt_arts")
    parser.add_argument("--download-dir", default="downloaded_cards")
    parser.add_argument("--output", default="netrunner-catalog.npz")
    parser.add_argument("--format", default="standard")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-alt-arts", action="store_true")
    parser.add_argument("--append", action="store_true", help="Append only new filename-stem IDs to an existing catalog.")
    parser.add_argument("--replace-existing", action="store_true", help="With --append, replace embeddings for existing IDs.")
    args = parser.parse_args()
    build_catalog(source=args.source, image_dir=args.image_dir, alt_art_dir=None if args.no_alt_arts else args.alt_art_dir, download_dir=args.download_dir, output_path=args.output, format_id=args.format, force_download=args.force_download, append=args.append, replace_existing=args.replace_existing)

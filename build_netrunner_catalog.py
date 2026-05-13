from netrunner_scanner.build_catalog import build_catalog

if __name__ == "__main__":
    build_catalog(
        image_dir="cards",
        output_path="netrunner-catalog.npz",
    )

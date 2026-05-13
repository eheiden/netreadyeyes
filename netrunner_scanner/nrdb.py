import re
from pathlib import Path

import requests

API_BASE = "https://api.netrunnerdb.com/api/v3/public"
IMAGE_BASE = "https://card-images.netrunnerdb.com/v2"


def sanitize_title(name):
    safe = re.sub(r"[\\/:*?'\"<>|()\[\],-.!]+", "_", name)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_")
    return safe.lower()


def nrdb_get(url, params=None):
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_active_card_pool_id(format_id="standard"):
    data = nrdb_get(f"{API_BASE}/formats")

    for item in data.get("data", []):
        if item.get("id") == format_id:
            attrs = item.get("attributes", {})
            pool_id = attrs.get("active_card_pool_id")

            if not pool_id:
                raise RuntimeError(f"Format {format_id!r} has no active_card_pool_id.")

            print(f"Using active {format_id} card pool: {pool_id}")
            return pool_id

    raise RuntimeError(f"Could not find NRDB format: {format_id}")


def fetch_card_pool(pool_id):
    return nrdb_get(f"{API_BASE}/card_pools/{pool_id}")


def extract_card_ids_from_card_pool(card_pool_json):
    data = card_pool_json.get("data", {})
    attrs = data.get("attributes", {})
    rels = data.get("relationships", {})

    relationship_cards = rels.get("cards", {}).get("data", [])
    card_ids = [
        item.get("id")
        for item in relationship_cards
        if item.get("id")
    ]

    if card_ids:
        return sorted(set(card_ids))

    for key in ["card_ids", "cards"]:
        values = attrs.get(key)
        if isinstance(values, list):
            return sorted(set(str(v) for v in values))

    # Some NRDB deployments provide card ids under relationships as links only.
    # If that happens, fetch the relationship URL.
    related = rels.get("cards", {}).get("links", {}).get("related")
    if related:
        related_json = nrdb_get(related)
        ids = [
            item.get("id")
            for item in related_json.get("data", [])
            if item.get("id")
        ]
        if ids:
            return sorted(set(ids))

    raise RuntimeError(
        "Could not find card ids in card pool response. "
        "Save/inspect the NRDB response and adjust extract_card_ids_from_card_pool()."
    )


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_cards_by_ids(card_ids):
    cards = []

    for chunk in chunked(card_ids, 100):
        params = {
            "fields[cards]": "stripped_title,title,printing_ids,faces",
            "filter[id]": ",".join(chunk),
        }

        data = nrdb_get(f"{API_BASE}/cards", params=params)
        cards.extend(data.get("data", []))

    return cards


def make_records_from_cards(cards, include_all_printings=True, include_faces=True):
    records = {}

    for card in cards:
        attrs = card.get("attributes", {})

        stripped_title = attrs.get("stripped_title") or attrs.get("title") or card.get("id")
        safe_title = sanitize_title(stripped_title)

        printing_ids = attrs.get("printing_ids", [])

        if not printing_ids:
            printing_ids = [card.get("id")]

        if include_all_printings:
            pids_to_include = printing_ids
        else:
            pids_to_include = [printing_ids[0]]

        for pid in pids_to_include:
            if not pid:
                continue

            name = f"{safe_title}_{pid}"

            records[name] = {
                "medium": f"{IMAGE_BASE}/medium/{pid}.jpg",
                "xlarge": f"{IMAGE_BASE}/xlarge/{pid}.webp",
                "large": f"{IMAGE_BASE}/large/{pid}.jpg",
                "source": "nrdb",
            }

        if include_faces:
            faces = attrs.get("faces") or []

            for face in faces:
                face_title = (
                    face.get("stripped_title")
                    or face.get("title")
                    or f"{stripped_title}_{face.get('index', 0)}"
                )

                safe_face_title = sanitize_title(face_title)
                face_images = face.get("images", {}).get("nrdb_classic", {})

                face_medium = face_images.get("medium") or face_images.get("small")
                face_xlarge = face_images.get("xlarge") or face_images.get("large")

                if not face_medium and not face_xlarge:
                    continue

                for pid in pids_to_include:
                    if not pid:
                        continue

                    face_name = f"{safe_face_title}_{pid}"

                    records[face_name] = {
                        "medium": face_medium,
                        "xlarge": face_xlarge,
                        "large": face_xlarge or face_medium,
                        "source": "nrdb_face",
                    }

    return records


def fetch_standard_records(format_id="standard"):
    pool_id = get_active_card_pool_id(format_id)
    pool_json = fetch_card_pool(pool_id)
    card_ids = extract_card_ids_from_card_pool(pool_json)

    print(f"Found {len(card_ids)} distinct card ids in {format_id} pool.")

    cards = fetch_cards_by_ids(card_ids)

    print(f"Fetched {len(cards)} card records from NRDB.")

    return make_records_from_cards(
        cards,
        include_all_printings=True,
        include_faces=True,
    )


def download_file(url, output_path, force=False):
    output_path = Path(output_path)

    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with output_path.open("wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)

    return output_path


def download_record_image(record_name, record, download_dir, force=False):
    download_dir = Path(download_dir)
    output_path = download_dir / f"{record_name}.jpg"

    if output_path.exists() and not force:
        return output_path

    urls_to_try = [
        record.get("xlarge"),
        record.get("large"),
        record.get("medium"),
    ]

    last_error = None

    for url in urls_to_try:
        if not url:
            continue

        try:
            return download_file(url, output_path, force=True)
        except Exception as e:
            last_error = e
            print(f"Failed URL for {record_name}: {url}")
            print(e)

    raise RuntimeError(f"Could not download any image for {record_name}: {last_error}")


def download_records(records, download_dir="downloaded_cards", force=False):
    downloaded = []

    for index, (record_name, record) in enumerate(sorted(records.items()), start=1):
        print(f"[{index}/{len(records)}] Downloading {record_name}")

        try:
            path = download_record_image(
                record_name=record_name,
                record=record,
                download_dir=download_dir,
                force=force,
            )

            downloaded.append(path)

        except Exception as e:
            print(f"FAILED: {record_name}")
            print(e)

    return downloaded

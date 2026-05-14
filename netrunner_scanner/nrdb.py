import re
from pathlib import Path
import requests
API_BASE="https://api.netrunnerdb.com/api/v3/public"; IMAGE_BASE="https://card-images.netrunnerdb.com/v2"
def sanitize_title(name):
    safe=re.sub(r"[\\/:*?'\"<>|()\[\],-.!]+","_",name); safe=re.sub(r"\s+","_",safe); safe=re.sub(r"_+","_",safe); return safe.strip("_").lower()
def nrdb_get(url,params=None):
    r=requests.get(url,params=params,timeout=30); r.raise_for_status(); return r.json()
def get_active_card_pool_id(format_id="standard"):
    data=nrdb_get(f"{API_BASE}/formats")
    for item in data.get("data",[]):
        if item.get("id")==format_id:
            pool=item.get("attributes",{}).get("active_card_pool_id")
            if not pool: raise RuntimeError(f"Format {format_id!r} has no active_card_pool_id.")
            print(f"Using active {format_id} card pool: {pool}"); return pool
    raise RuntimeError(f"Could not find NRDB format: {format_id}")
def fetch_card_pool(pool_id): return nrdb_get(f"{API_BASE}/card_pools/{pool_id}")
def extract_card_ids_from_card_pool(j):
    data=j.get("data",{}); attrs=data.get("attributes",{}); rels=data.get("relationships",{})
    ids=[x.get("id") for x in rels.get("cards",{}).get("data",[]) if x.get("id")]
    if ids: return sorted(set(ids))
    for key in ["card_ids","cards"]:
        vals=attrs.get(key)
        if isinstance(vals,list): return sorted(set(str(v) for v in vals))
    related=rels.get("cards",{}).get("links",{}).get("related")
    if related:
        r=nrdb_get(related); ids=[x.get("id") for x in r.get("data",[]) if x.get("id")]
        if ids: return sorted(set(ids))
    raise RuntimeError("Could not find card ids in card pool response.")
def chunked(items,size):
    for i in range(0,len(items),size): yield items[i:i+size]
def fetch_cards_by_ids(card_ids):
    cards=[]
    for chunk in chunked(card_ids,100):
        data=nrdb_get(f"{API_BASE}/cards",params={"fields[cards]":"stripped_title,title,printing_ids,faces","filter[id]":",".join(chunk)})
        cards.extend(data.get("data",[]))
    return cards
def make_records_from_cards(cards,include_all_printings=True,include_faces=True):
    records={}
    for card in cards:
        attrs=card.get("attributes",{}); title=attrs.get("stripped_title") or attrs.get("title") or card.get("id"); safe=sanitize_title(title)
        pids=attrs.get("printing_ids",[]) or [card.get("id")]; pids_to_include=pids if include_all_printings else [pids[0]]
        for pid in pids_to_include:
            if pid: records[f"{safe}_{pid}"]={"medium":f"{IMAGE_BASE}/medium/{pid}.jpg","xlarge":f"{IMAGE_BASE}/xlarge/{pid}.webp","large":f"{IMAGE_BASE}/large/{pid}.jpg","source":"nrdb"}
        if include_faces:
            for face in attrs.get("faces") or []:
                ft=face.get("stripped_title") or face.get("title") or f"{title}_{face.get('index',0)}"; sf=sanitize_title(ft); imgs=face.get("images",{}).get("nrdb_classic",{})
                med=imgs.get("medium") or imgs.get("small"); xl=imgs.get("xlarge") or imgs.get("large")
                if not med and not xl: continue
                for pid in pids_to_include:
                    if pid: records[f"{sf}_{pid}"]={"medium":med,"xlarge":xl,"large":xl or med,"source":"nrdb_face"}
    return records
def fetch_standard_records(format_id="standard"):
    ids=extract_card_ids_from_card_pool(fetch_card_pool(get_active_card_pool_id(format_id)))
    print(f"Found {len(ids)} distinct card ids in {format_id} pool.")
    cards=fetch_cards_by_ids(ids); print(f"Fetched {len(cards)} card records from NRDB.")
    return make_records_from_cards(cards,True,True)
def download_file(url,output_path,force=False):
    output_path=Path(output_path)
    if output_path.exists() and not force: return output_path
    output_path.parent.mkdir(parents=True,exist_ok=True); r=requests.get(url,stream=True,timeout=60); r.raise_for_status()
    with output_path.open("wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)
    return output_path
def download_record_image(record_name,record,download_dir,force=False):
    out=Path(download_dir)/f"{record_name}.jpg"
    if out.exists() and not force: return out
    last=None
    for url in [record.get("xlarge"),record.get("large"),record.get("medium")]:
        if not url: continue
        try: return download_file(url,out,True)
        except Exception as e: last=e; print(f"Failed URL for {record_name}: {url}\n{e}")
    raise RuntimeError(f"Could not download any image for {record_name}: {last}")
def download_records(records,download_dir="downloaded_cards",force=False):
    out=[]
    for i,(name,rec) in enumerate(sorted(records.items()),1):
        print(f"[{i}/{len(records)}] Downloading {name}")
        try: out.append(download_record_image(name,rec,download_dir,force))
        except Exception as e: print(f"FAILED: {name}\n{e}")
    return out

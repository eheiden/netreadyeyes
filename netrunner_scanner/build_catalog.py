from pathlib import Path
import re
import numpy as np
from PIL import Image
import collector_vision as cvg
from .nrdb import fetch_standard_records, download_records
SUPPORTED_EXTENSIONS={".jpg",".jpeg",".png",".webp"}
def load_embedder():
    print("Loading CollectorVision embedder...")
    return cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg").embedder
def iter_image_files(image_dir):
    image_dir=Path(image_dir)
    if not image_dir.exists(): return []
    return [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
def normalize_alt_art_filename(path):
    stem=path.stem.lower(); stem=re.sub(r'[\\/:*?"<>|]',"",stem); stem=re.sub(r"[“”‘’]","",stem); stem=re.sub(r"\s+","_",stem); stem=re.sub(r"_+","_",stem); return stem.strip("_")
def collect_folder_images(image_dir): return [(p.stem,p) for p in iter_image_files(image_dir)]
def collect_alt_art_images(alt_art_dir):
    alt_art_dir=Path(alt_art_dir)
    if not alt_art_dir.exists(): print(f"No alt-art folder found at {alt_art_dir}; skipping alt arts."); return []
    images=[]
    for p in iter_image_files(alt_art_dir):
        stem=normalize_alt_art_filename(p)
        if "_alt_" not in stem: print(f"Skipping alt-art without '_alt_' in filename: {p.name}"); continue
        images.append((stem,p))
    print(f"Found {len(images)} alt-art image(s)."); return images
def collect_nrdb_images(download_dir,format_id="standard",force_download=False):
    records=fetch_standard_records(format_id=format_id)
    print(f"NRDB produced {len(records)} image record(s), including all printings and faces.")
    return [(p.stem,p) for p in download_records(records=records,download_dir=download_dir,force=force_download)]
def load_existing_catalog(catalog_path):
    catalog_path=Path(catalog_path)
    if not catalog_path.exists(): return [], None
    data=np.load(catalog_path,allow_pickle=True)
    return [str(x) for x in data["ids"]], data["embeddings"]
def save_catalog(output_path,ids,embeddings):
    np.savez_compressed(output_path,ids=np.array(ids),embeddings=np.array(embeddings))
def embed_images(images,output_path,append=False,replace_existing=False):
    output_path=Path(output_path)
    if not images: raise RuntimeError("No images found to embed.")
    existing_ids=[]; existing_embeddings=None
    if append:
        existing_ids, existing_embeddings=load_existing_catalog(output_path)
        if existing_ids: print(f"Loaded existing catalog: {output_path}\nExisting IDs: {len(existing_ids)}")
        else: print(f"Append requested, but no existing catalog was found at {output_path}. A new catalog will be created.")
    existing_id_set=set(existing_ids)
    if append and not replace_existing:
        images_to_embed=[(cid,p) for cid,p in images if cid not in existing_id_set]
        skipped=len(images)-len(images_to_embed)
        if skipped: print(f"Skipping {skipped} image(s) already present in catalog.")
    else: images_to_embed=images
    if not images_to_embed: print("No new images to embed."); return
    embedder=load_embedder(); new_ids=[]; new_embeddings=[]; seen=set()
    print(f"Embedding {len(images_to_embed)} image(s).\nOutput file: {output_path}")
    for i,(cid,p) in enumerate(images_to_embed,1):
        if cid in seen: print(f"Skipping duplicate id in this run: {cid}"); continue
        seen.add(cid); print(f"[{i}/{len(images_to_embed)}] Embedding {cid}")
        try:
            new_ids.append(cid); new_embeddings.append(embedder.embed(Image.open(p).convert("RGB")))
        except Exception as e: print(f"FAILED: {p}\n{e}")
    if not new_embeddings: print("No embeddings were created."); return
    if append and existing_ids and existing_embeddings is not None:
        if replace_existing:
            repl=dict(zip(new_ids,new_embeddings)); final_ids=[]; final_embeddings=[]; replaced=0
            for cid,emb in zip(existing_ids,existing_embeddings):
                if cid in repl: final_ids.append(cid); final_embeddings.append(repl.pop(cid)); replaced+=1
                else: final_ids.append(cid); final_embeddings.append(emb)
            added=len(repl)
            for cid,emb in repl.items(): final_ids.append(cid); final_embeddings.append(emb)
            print(f"Replaced existing IDs: {replaced}\nAdded new IDs: {added}")
        else:
            final_ids=existing_ids+new_ids; final_embeddings=list(existing_embeddings)+new_embeddings; print(f"Added new IDs: {len(new_ids)}")
    else: final_ids=new_ids; final_embeddings=new_embeddings
    save_catalog(output_path,final_ids,final_embeddings); print(f"\nSaved catalog: {output_path}\nTotal cards embedded: {len(final_ids)}")
def build_catalog(source="nrdb",image_dir="cards",alt_art_dir="alt_arts",download_dir="downloaded_cards",output_path="netrunner-catalog.npz",format_id="standard",force_download=False,append=False,replace_existing=False):
    if source=="nrdb": images=collect_nrdb_images(download_dir,format_id,force_download)
    elif source=="folder": images=collect_folder_images(image_dir)
    else: raise ValueError(f"Unknown catalog source: {source}")
    if alt_art_dir: images.extend(collect_alt_art_images(alt_art_dir))
    embed_images(images,Path(output_path),append,replace_existing)

# CollectorVision Netrunner Scanner

## Append a few local cards to an existing catalog

Put the new images in a folder, for example:

```text
extra_cards/
  my_missing_card_12345.jpg
  custom_proxy_alt_stream.jpg
```

Then run:

```powershell
python .\build_netrunner_catalog.py --source folder --image-dir extra_cards --append
```

This loads the existing `netrunner-catalog.npz`, embeds only images whose filename stem is not already present, and saves the updated catalog.

To replace existing embeddings for matching filename stems:

```powershell
python .\build_netrunner_catalog.py --source folder --image-dir extra_cards --append --replace-existing
```

## Build from NRDB Standard

```powershell
python .\build_netrunner_catalog.py --source nrdb --format standard
```

## Build from a local folder

```powershell
python .\build_netrunner_catalog.py --source folder --image-dir cards
```

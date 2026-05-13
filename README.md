# CollectorVision Netrunner Scanner Refactor

## Build the catalog

Put card images in:

```text
cards/
```

Example:

```text
cards/buzzsaw_30005.jpg
cards/sure_gamble_30030.jpg
```

Then run:

```powershell
python .\build_netrunner_catalog.py
```

This creates:

```text
netrunner-catalog.npz
```

The filename stem becomes the card ID.

## Run the scanner

```powershell
python .\live_scanner.py
```

## Controls

```text
Mouse drag inside box = move ROI
Mouse drag edge/corner = resize ROI
Q = manually scan LEFT / pink playmat
E = manually scan RIGHT / blue playmat
S = save ROI settings
L = load ROI settings
R = reset ROIs to left/right split
ESC = quit
```

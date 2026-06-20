# Cityscapes Dataset Setup

## Important

Cityscapes is not a fully open direct-download dataset. The official website requires a user account for downloading the data. Do not commit usernames, passwords, cookies, downloaded ZIP files, or extracted dataset files to GitHub.

Official website:

https://www.cityscapes-dataset.com/

## Required Files

For this project you need:

- `gtFine_trainvaltest.zip`
- `leftImg8bit_trainvaltest.zip`

After extraction, the expected structure is:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Option 1: Helper Script

From the project root:

```bash
chmod +x scripts/download_cityscapes.sh
./scripts/download_cityscapes.sh
```

The script installs the official `cityscapesscripts` package and calls `csDownload` for the required ZIP files.

## Option 2: Manual Download

1. Register or log in to the Cityscapes website.
2. Download:
   - `gtFine_trainvaltest.zip`
   - `leftImg8bit_trainvaltest.zip`
3. Move both files into `data/cityscapes/`.
4. Extract them:

```bash
cd data/cityscapes
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
```

## Quick Verification

From the repository root, run:

```bash
find data/cityscapes/leftImg8bit/train -name '*_leftImg8bit.png' | head
find data/cityscapes/gtFine/train -name '*_gtFine_labelIds.png' | head
```

Then test one training epoch:

```bash
python train.py --data-root data/cityscapes --epochs 1 --batch-size 1 --image-size 128
```

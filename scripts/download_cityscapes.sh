#!/usr/bin/env bash
set -e

echo "Cityscapes dataset setup"
echo "This script uses the official cityscapesscripts downloader."
echo "You need a valid Cityscapes account before running it."
echo "Create an account at: https://www.cityscapes-dataset.com/"
echo ""

python3 -m pip install --upgrade cityscapesscripts

mkdir -p data/cityscapes
cd data/cityscapes

echo "Downloading gtFine annotations..."
csDownload gtFine_trainvaltest.zip

echo "Downloading leftImg8bit images..."
csDownload leftImg8bit_trainvaltest.zip

echo "Extracting files..."
unzip -q gtFine_trainvaltest.zip
unzip -q leftImg8bit_trainvaltest.zip

echo "Cityscapes dataset is ready in data/cityscapes"

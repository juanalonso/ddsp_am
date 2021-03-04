#!/bin/bash
set -ex

#orig_dir=$(pwd)
#tmp_dir=$(mktemp -d -t ddsp-XXXX)
#git clone https://github.com/magenta/ddsp.git $tmp_dir
#cd $tmp_dir

python setup.py sdist
python setup.py bdist_wheel --universal

rm -rf build
rm -rf ddsp.egg-info
rm -rf .eggs
rm -rf ddsp/__pycache__

cp dist/ddsp* /Volumes/GoogleDrive/Mi\ unidad/SMC\ 09/DDSP/dist

#cd $orig_dir
#rm -rf $tmp_dir

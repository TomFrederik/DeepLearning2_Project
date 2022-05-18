#!/bin/bash

URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

# get script dir
DIR=$(dirname "$SCRIPT")"/data"
echo "Downloading data into $DIR..."

# make sure the dir exists
mkdir -p $DIR

# download imagenette2
wget -O $DIR/imagenette2.tgz $URL

# extract imagenette2
tar -xzf $DIR/imagenette2.tgz -C $DIR/

# remove imagenette2.tgz
rm $DIR/imagenette2.tgz
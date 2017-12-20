#!/bin/bash

for i in $(seq $1 $2); do
    montage -geometry 640x320+3+3 -border 1 -tile 2x3 -background "rgb(32,32,32)" output_images/$i-*.png montage_dir/$i.png
done
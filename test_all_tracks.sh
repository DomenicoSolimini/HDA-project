#!/bin/bash

FILES=data/gt_tracks/*

for f in $FILES
do
  TRACK=$(echo $f | sed -n "s/^.*_\(\S*_.\)\.npy/\1/p")
  python3 GT-tracks_filtering.py --track "$TRACK"
done

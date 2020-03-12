#!/bin/bash

FILES=data/gt_tracks/*

# Test if the previous metrics file exists and removes it
if test -f "metrics.txt"; then
  echo "Metrics file updated!"
  rm metrics.txt
fi

# Create a new metric storage file
touch metrics.txt

# Loop through the GT files
for f in $FILES
do
  # Extract the track name
  TRACK=$(echo $f | sed -n "s/^.*radar1trg_\(\S*_.\)\.npy/\1/p")

  # For all the non-empty names
  if test -n "$TRACK"; then

    echo "********"
    echo "Editing track $TRACK"
    echo "********"

    # Execute the test function and append results to metrics file
    python3 GT-tracks_filtering.py --track "$TRACK" >> metrics.txt
  fi

done

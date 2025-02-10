#!/bin/bash

declare -a llm_pos=("low-centrality")  # "high-centrality"
declare -a sentiment=("negative")  # "positive"

for pos in "${llm_pos[@]}"; do
  for s in "${sentiment[@]}"; do

    python3 sentiment-propagation.py -L "$pos" -T "referendum-$s"  # "brexit-$s"  #

  done
done
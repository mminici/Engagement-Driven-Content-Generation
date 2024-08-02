#!/bin/bash

# "high-degree"
# "high-centrality"

# ABBIAMO:
# low-degree positive
# high-degree/high-centrality positive (sono uguali)
# high-degree negative
# high-centrality negative

# Quindi VANNO FATTI:
# low-centrality negative (1) Fatta
# low-centrality positive (2) DA RILANCIARE
# low-degree negative (3) Fatta
declare -a llm_pos=("low-centrality")
# ("high-degree" "low-degree" "high-centrality" "low-centrality")
declare -a sentiment=("positive")
# ("positive" "negative")

for pos in "${llm_pos[@]}"; do
  for s in "${sentiment[@]}"; do

    if ! ( ([ $pos = "low-degree" ]) && ([ $s = "positive" ]) ); then
      python3 sentiment-propagation.py -L "$pos" -T "brexit-$s"
    fi

  done
done
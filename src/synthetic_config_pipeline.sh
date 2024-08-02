#!/bin/bash
declare -a network_opinions=("positive" "negative" "neutral" "uniform")
#declare -a network_opinions=("neutral")

declare -a llm_pos=("comm-smallest" "central" "comm-largest" "echo-low" "echo-high")
declare -a modularity=("high" "low")
declare -a homophily=("high" "low")


for network_opinion in "${network_opinions[@]}"; do

  for pos in "${llm_pos[@]}"; do

    for mod in "${modularity[@]}"; do

      for hom in "${homophily[@]}"; do

#        if ([ $pos = "central" ]) ||
#           ([ $pos = "comm-largest" ] && [ $mod = "low" ] && [ $hom = "low" ]) ||
#           ([ $pos = "comm-smallest" ] && ! ([ $mod = "high" ] && [ $hom = "high" ])); then
#          echo $pos $mod $hom
          python3 sentiment-propagation.py -O "$network_opinion" -L "$pos" -M "$mod" -H "$hom" -T "cats"
#        fi

      done
    done
  done
done
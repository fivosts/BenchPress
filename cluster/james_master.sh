#!/bin/bash

# Let's stick to the following allocation:
# james06 -> db0
# james09 -> db1
# james10 -> db2
# james11 -> db3
# james13 -> db4
# james15 -> db5
# james16 -> db6

declare -A nodes
nodes=( ["james06.inf.ed.ac.uk"]="0" ["james09.inf.ed.ac.uk"]="1" ["james10.inf.ed.ac.uk"]="2" ["james11.inf.ed.ac.uk"]="3" ["james13.inf.ed.ac.uk"]="4" ["james15.inf.ed.ac.uk"]="5" ["james16.inf.ed.ac.uk"]="6")

if [ $# -ge 1 ] && [ "$1" == "--kill" ]; then
  for key in ${!nodes[@]}; do
    ssh s1879742@${key} "kill $(pgrep clgen)"
  done
elif [ $# -ge 1 ] && [ "$1" == "--clean_workspace" ]; then
  for key in ${!nodes[@]}; do
    ssh s1879742@${key} "rm -rf /disk/scratch/s1879742/c_corpus_pretrain"
  done
elif [ $# -ge 1 ] && [ "$1" == "--dispatch" ]; then
  for key in ${!nodes[@]}; do
    ssh s1879742@${key} "bash -s" < ./run_james.sh&
  done
else
  echo "No option selected."
fi

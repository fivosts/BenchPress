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
nodes=( ["james06.inf.ed.ac.uk"]="0" ["james09.inf.ed.ac.uk"]="1" ["james10.inf.ed.ac.uk"]="2" ["james11.inf.ed.ac.uk"]="3" ["james13.inf.ed.ac.uk"]="4" ["anne.inf.ed.ac.uk"]="4" ["mary.inf.ed.ac.uk"]="5" ["james15.inf.ed.ac.uk"]="5" ["james16.inf.ed.ac.uk"]="6")

if [ ! -f /disk/scratch/s1879742/bigQuery/clgen_c_github_"${nodes[$(hostname)]}".db ]; then
  echo "Database "${nodes[$(hostname)]}" for $(hostname) not found!"
  exit
fi

cp /disk/scratch/s1879742/bigQuery/clgen_c_github_"${nodes[$(hostname)]}".db /disk/scratch/s1879742/bigQuery/clgen_c_github.db

cd
cd clgen

stdbuf -oL ./clgen --local_filesystem /disk/scratch/s1879742/tmp --stop_after corpus --workspace_dir /disk/scratch/s1879742/c_corpus_pretrain --config model_zoo/BERT/bert_base_pretrain.pbtxt &> log-$(hostname).txt

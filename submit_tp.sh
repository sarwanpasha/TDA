#!/bin/bash
# Submit every TPAMI worklist. Jobs that need the prerequisite graphs wait for it.
set -e
cd $HOME/tda_imc/code
J=$HOME/tda_imc/jobs/tp
sub() { sbatch --parsable --job-name=TDA_$1 "${@:3}" job_tp.sbatch $J/$1.tsv $2; }
PREP=$(sub prep 3)
echo prep $PREP
for l in alpha_lip curves robust graphnets perslay masks codex20 scaling transfer gallery; do
  n=30; [ $l = robust ] && n=20; [ $l = scaling ] && n=1
  echo $l $(sub $l $n)
done
for l in jackson cap2000 cap5000; do
  echo $l $(sub $l 30 --dependency=afterok:$PREP)
done
squeue -u $USER -o "%.10i %.20j %.12P %.8T %.10M %.6D %R"

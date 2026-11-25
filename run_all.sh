#!/bin/sh
# Every command behind the numbers in the stage descriptions, on a laptop CPU. Usage: ./run_all.sh [seed]
set -e
SEED=${1:-0}
P=solutions/python/18-message-passing/code/your_program.sh
OUT=runs/seed$SEED
mkdir -p "$OUT"
if [ "$SEED" = 0 ]; then
  $P train-gpt data/tinyshakespeare.txt --steps 2000 --out-dir "$OUT"
fi
$P train-looped --seed "$SEED" --out-dir "$OUT"
$P injection --seed "$SEED" --kinds looped,looped_none,looped_message --out-dir "$OUT"
$P compare --seed "$SEED" --out-dir "$OUT"
for c in logit-lens tuned-lens probe patch ablate sae exit; do
  $P $c --seed "$SEED" --ckpt "$OUT/looped_s$SEED.pt" --out-dir "$OUT"
done
# the same lenses and probes on the two other ways of looping (stage 12 compares them)
for kind in looped_none looped_message; do
  for c in logit-lens tuned-lens probe; do
    $P $c --seed "$SEED" --kind $kind --ckpt "$OUT/${kind}_s$SEED.pt" --out-dir "$OUT"
  done
done

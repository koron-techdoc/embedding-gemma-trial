#!env bash

set -eu

export PYTHONUTF8=1

OUTDIR=./EmbeddingGemma
MODEL=google/embeddinggemma-300m
PROMPT=Clustering
BATCHSIZE=25

TRUTH="./truth_full.txt"
ACCURACY_000="${OUTDIR}/accuracy-trained-000.txt"
TRAIN_DATA="${OUTDIR}/training_data.tsv"
MODEL_OUTDIR="${OUTDIR}/trained_model"
ACCURACY_020="${OUTDIR}/accuracy-trained-020.txt"
ACCURACY_040="${OUTDIR}/accuracy-trained-040.txt"
ACCURACY_060="${OUTDIR}/accuracy-trained-060.txt"
ACCURACY_080="${OUTDIR}/accuracy-trained-080.txt"
ACCURACY_100="${OUTDIR}/accuracy-trained-100.txt"
RESULT_CASTLE_000="${OUTDIR}/result-castle-000.txt"
RESULT_CASTLE_100="${OUTDIR}/result-castle-100.txt"
RESULT_OLDCOUNTRY_000="${OUTDIR}/result-oldcountry-000.txt"
RESULT_OLDCOUNTRY_100="${OUTDIR}/result-oldcountry-100.txt"
RESULT_JRSTATION_000="${OUTDIR}/result-jrstation-000.txt"
RESULT_JRSTATION_100="${OUTDIR}/result-jrstation-100.txt"
RESULT_JREKI_000="${OUTDIR}/result-jreki-000.txt"
RESULT_JREKI_100="${OUTDIR}/result-jreki-100.txt"

if [ ! -d "${OUTDIR}" ] ; then
  mkdir -p "${OUTDIR}"
fi

clustering() {
  model=$1
  prompt=$2
  truth=$3
  result=$4
  if [ ! -f "$result" ] ; then
    echo "clustering: model=$model prompt=$prompt truth=$truth result=$result"
    ./cluster_local_govs.py -m "$model" -k "$prompt" -l "$truth" > "$result"
  fi
}

# 1. First clustering without training.
if [ ! -f "${ACCURACY_000}" ] ; then
  clustering "$MODEL" "$PROMPT" "$TRUTH" "$ACCURACY_000"
fi

# 2. Generate training data.
if [ ! -f "${TRAIN_DATA}" ] ; then
  go run ./gen_train.go "${ACCURACY_000}" > "${TRAIN_DATA}"
fi

TRAINED_020="${MODEL_OUTDIR}/checkpoint-77"
TRAINED_040="${MODEL_OUTDIR}/checkpoint-154"
TRAINED_060="${MODEL_OUTDIR}/checkpoint-231"
TRAINED_080="${MODEL_OUTDIR}/checkpoint-308"
TRAINED_100="${MODEL_OUTDIR}/checkpoint-385"

# 3. Execute training
if [ ! -d "${MODEL_OUTDIR}" ] ; then
  ./train.py -m "${MODEL}" -k "${PROMPT}" -t "${TRAIN_DATA}" -o "${MODEL_OUTDIR}" -b "$BATCHSIZE"
fi

# 4. Evaluate trained model.
clustering "$TRAINED_020" "$PROMPT" "$TRUTH" "$ACCURACY_020"
clustering "$TRAINED_040" "$PROMPT" "$TRUTH" "$ACCURACY_040"
clustering "$TRAINED_060" "$PROMPT" "$TRUTH" "$ACCURACY_060"
clustering "$TRAINED_080" "$PROMPT" "$TRUTH" "$ACCURACY_080"
clustering "$TRAINED_100" "$PROMPT" "$TRUTH" "$ACCURACY_100"

# 5. Clustering variants
clustering "$MODEL"       "$PROMPT" ./japan_castles.tsv "$RESULT_CASTLE_000"
clustering "$TRAINED_100" "$PROMPT" ./japan_castles.tsv "$RESULT_CASTLE_100"

clustering "$MODEL"       "$PROMPT" ./japan_old_countries.tsv "$RESULT_OLDCOUNTRY_000"
clustering "$TRAINED_100" "$PROMPT" ./japan_old_countries.tsv "$RESULT_OLDCOUNTRY_100"

clustering "$MODEL"       "$PROMPT" ./jr_stations.tsv "$RESULT_JRSTATION_000"
clustering "$TRAINED_100" "$PROMPT" ./jr_stations.tsv "$RESULT_JRSTATION_100"

clustering "$MODEL"       "$PROMPT" ./jr_stations_eki.tsv "$RESULT_JREKI_000"
clustering "$TRAINED_100" "$PROMPT" ./jr_stations_eki.tsv "$RESULT_JREKI_100"

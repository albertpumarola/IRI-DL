#!/usr/bin/env bash
DIR=" $(dirname cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export PYTHONPATH=$DIR:$PYTHONPATH
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=$1
source ~/miniconda3/bin/activate IRI-DL
#!/usr/bin/env bash
ROOT_PATH=$(dirname "$(dirname "$0")")
export PYTHONPATH=$ROOT_PATH:$PYTHONPATH

# LANG=python
MODEL=python

cd $ROOT_PATH


onmt-main \
    --config models/$MODEL/dc.yml \
    --auto_config \
    infer \
    --features_file input_ast.txt input_code.txt \
    --predictions_file result.txt

#!/usr/bin/env bash
ROOT_PATH=$(dirname "$(dirname "$0")")
export PYTHONPATH=$ROOT_PATH:$PYTHONPATH

# LANG=python
MODEL=ruby

cd $ROOT_PATH

onmt-main \
    --config /models/$MODEL/data1.yml \
    --auto_config \
    infer \
    --features_file datasets/ruby/test/src_path.txt datasets/ruby/test/src_code.txt \
    --predictions_file models/$MODEL/predictions_maml1.txt


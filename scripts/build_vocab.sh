#! /bin/bash
ROOT_PATH=$(dirname "$(dirname "$0")")
export PYTHONPATH=$ROOT_PATH:$PYTHONPATH

LANG=java_v2

cd $ROOT_PATH

# CUDA_VISIBLE_DEVICES=0 onmt-build-vocab --size 50000 --save_vocab datasets/$LANG/vocab/src-vocab.txt datasets/$LANG/train/src_code.csv 
# CUDA_VISIBLE_DEVICES=0 onmt-build-vocab --size 8000 --save_vocab datasets/$LANG/vocab/tgt-vocab.txt datasets/$LANG/train/tgt_title.csv
onmt-build-vocab --size 5000 --save_vocab datasets/vocab/path-vocab.txt datasets/pythonFull/train/src_path.txt datasets/javaFull/train/src_path.txt
onmt-build-vocab --size 500000 --save_vocab datasets/vocab/src-vocab.txt datasets/pythonFull/train/src_code.txt datasets/javaFull/train/src_code.txt
onmt-build-vocab --size 500000 --save_vocab datasets/vocab/tgt-vocab.txt datasets/pythonFull/train/tgt_title.txt datasets/javaFull/train/tgt_title.txt 

# onmt-build-vocab --size 25000 --save_vocab datasets/$LANG/vocab/src-vocab.txt datasets/$LANG/train/src_code.csv 
# onmt-build-vocab --size 10000 --save_vocab datasets/$LANG/vocab/tgt-vocab.txt datasets/$LANG/train/tgt_title.csv
# onmt-build-vocab --size 150000 --save_vocab datasets/vocab/path-vocab.txt datasets/python_big/train/src_path.csv datasets/python_big/test/src_path.csv datasets/python_big/eval/src_path.csv datasets/java/train/src_path.csv datasets/java/eval/src_path.csv

# onmt-build-vocab --from_vocab datasets/python/vocab/code_pieces.vocab --from_format sentencepiece --save_vocab datasets/python/vocab/code_pieces.csv
# onmt-build-vocab --from_vocab datasets/python/vocab/title_pieces.vocab --from_format sentencepiece --save_vocab datasets/python/vocab/title_pieces.csv

# onmt-build-vocab --size 100 --save_vocab datasets/$LANG/vocab/ast-vocab.txt datasets/$LANG/train/src_path.csv 

#!/bin/bash

SESSION=base
MODEL=python

#---------------->>>下方勿动<<<<------------------------------------------------------------
ENV_NAME=gytf
ROOT_PATH=$(dirname "$(dirname "$0")")
if [[ "$(tmux ls | grep $SESSION:)" == "" ]]
then
    tmux new -d -s $SESSION
    tmux send -t $SESSION "conda activate $ENV_NAME" ENTER
    tmux send -t $SESSION "export PYTHONPATH=$ROOT_PATH:$PYTHONPATH" ENTER
fi

tmux send -t $SESSION "cd $ROOT_PATH" ENTER
tmux send -t $SESSION "echo 'running'" ENTER

tmux send -t $SESSION "onmt-main --model  ../models/model.py \
                                 --gpu_allow_growth --config ../models/$MODEL.yml \
                                 --auto_config --checkpoint_path ./exps/python train --with_eval" ENTER
tmux send -t $SESSION "cd $ROOT_PATH" ENTER

tmux attach -t $SESSION

#tmux send -t $SESSION "onmt-main --model $ROOT_PATH/models/$MODEL/model.py \
#                                --gpu_allow_growth --config $ROOT_PATH/models/$MODEL/data.yml \
#                                 --auto_config train --with_eval" ENTER
#! /bin/bash

ENV_NAME=gytf
LANG=python


#---------------->>>下方勿动<<<<------------------------------------------------------------
__conda_setup="$('/opt/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
            . "/opt/anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate $ENV_NAME
ROOT_PATH=$(dirname "$(dirname "$0")")
export PYTHONPATH=$ROOT_PATH:$PYTHONPATH
cd $ROOT_PATH
echo 'running'


# onmt-main --model $ROOT_PATH/models/model.py \
#           --gpu_allow_growth --config $ROOT_PATH/models/$LANG.yml \
#           --auto_config train --with_eval

onmt-main --model ../models/model.py \
          --gpu_allow_growth --config ../models/$LANG.yml \
          --auto_config  train --with_eval

export CUDA_VISIBLE_DEVICES=${1:-0}
export HF_HUB_OFFLINE=True
SRC_PATH=${2:-'/public/KITTI360/'}
# SEQ=${3:-'0006'}
# START_FRAME=${4:-9038}
# END_FRAME=${5:-9223}
# SEQ=${3:-'0009'}
# START_FRAME=${4:-3972}
# END_FRAME=${5:-4258}
# SEQ=${3:-'0000'}
# START_FRAME=${4:-2501}
# END_FRAME=${5:-2706}
# SEQ=${3:-'0000'}
# START_FRAME=${4:-2913}
# END_FRAME=${5:-3233}
SEQ=${3:-'0000'}
START_FRAME=${4:-372}
END_FRAME=${5:-610}
NOTE=${6:-"test"}


# To note that the raw vegs pipeline contains both the lidar and colmap points as the gaussian position initialization.

HF_ENDPOINT=https://hf-mirror.com python ../train.py -s ${SRC_PATH} \
                --seq 2013_05_28_drive_${SEQ}_sync \
                --start_frame ${START_FRAME}\
                --end_frame ${END_FRAME} \
                --exp_note ${NOTE} \
                --save_results_as_images \
                --exclude_lidar \
                --output_dir output_${START_FRAME}_${END_FRAME}_prune_floaters \
                --cache_dir /home/$USER/.cache \
                --eval

# if no eval, whole images will be used as trainset, otherwise, select one image every 8 images
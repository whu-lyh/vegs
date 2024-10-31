export CUDA_VISIBLE_DEVICES=${1:-0}

# SOURCE_PATH=${2:-'/public/3D_gaussian_splatting_data/lego/'}
# EXP_NAME=${3:-'lego'}

# python ../train_raw.py  -s ${SOURCE_PATH} \
#                         --output_dir /workspace/WorkSpaceRec/vegs/bash_scripts/output_${EXP_NAME} \
#                         --exp_note prune_floater \
#                         --data_type blender \
#                         --iterations 30_000 \
#                         --eval

##############################################################################################################

# SOURCE_PATH=${2:-'/public/3D_gaussian_splatting_data/4seasons_sub1/'}
# EXP_NAME=${3:-'4seasons'}

# python ../train_raw.py  -s ${SOURCE_PATH} \
#                         --output_dir /public/3D_gaussian_splatting_data/4seasons_sub1/output_${EXP_NAME} \
#                         --exp_note car_mask \
#                         --data_type colmap \
#                         --iterations 50_000 \
#                         --eval

##############################################################################################################

# SOURCE_PATH=${2:-'/public/3D_gaussian_splatting_data/Shanghai/181013/sub2'}
# EXP_NAME=${3:-'Shanghai'}

# python ../train_raw.py  -s ${SOURCE_PATH} \
#                         --output_dir /public/3D_gaussian_splatting_data/Shanghai/181013/sub2/output/${EXP_NAME} \
#                         --exp_note lidar_car_mask_clean \
#                         --data_type MLS \
#                         --iterations 30_000 \
#                         --eval

# SOURCE_PATH=${2:-'/public/3D_gaussian_splatting_data/Shanghai/181013/sub1'}
# EXP_NAME=${3:-'Shanghai'}

# python ../train_raw.py  -s ${SOURCE_PATH} \
#                         --output_dir /public/3D_gaussian_splatting_data/Shanghai/181013/sub1/output/${EXP_NAME} \
#                         --exp_note lidar_car_mask_clean_sphere \
#                         --data_type MLS \
#                         --iterations 50_000 \
#                         --eval

#############################################################################################################

# SOURCE_PATH=${2:-'/public/3D_gaussian_splatting_data/wuhan_gs_test_41'}
# EXP_NAME=${3:-'Wuhan'}

# python ../train_raw.py  -s ${SOURCE_PATH} \
#                         --output_dir /public/3D_gaussian_splatting_data/wuhan_gs_test_41/output/${EXP_NAME} \
#                         --exp_note r3dgs_mask \
#                         --data_type MLS \
#                         --iterations 30_000

#############################################################################################################

# SOURCE_PATH=${2:-'/public/3D_gaussian_splatting_data/cz'}
# EXP_NAME=${3:-'Wuhan'}

# python ../train_raw.py  -s ${SOURCE_PATH} \
#                         --output_dir /public/3D_gaussian_splatting_data/cz/output/${EXP_NAME} \
#                         --exp_note r3dgs_mask \
#                         --data_type MLS_Raw \
#                         --resolution 2 \
#                         --iterations 30_000


SOURCE_PATH=${2:-'/public/3D_gaussian_splatting_data/cz_03'}
EXP_NAME=${3:-'Wuhan'}

python ../train_raw.py  -s ${SOURCE_PATH} \
                        --output_dir /public/3D_gaussian_splatting_data/cz_03/output/${EXP_NAME} \
                        --exp_note r3dgs_mask \
                        --data_type MLS_Raw \
                        --resolution 2 \
                        --iterations 30_000
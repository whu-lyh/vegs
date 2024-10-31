export CUDA_VISIBLE_DEVICES=${1:-0}

# MODELPATH=${2:-'/home/nas4_user/sungwonhwang/ws/3dgs_kitti/output_new/2013_05_28_drive_0000_sync_0000000372_0000000610/35e47f31-a_full_test'}

# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_372_610/2013_05_28_drive_0000_sync_0000000372_0000000610/28de535b-c_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_372_610_colmap/2013_05_28_drive_0000_sync_0000000372_0000000610/0917f14f-7_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_372_610_lidar/2013_05_28_drive_0000_sync_0000000372_0000000610/2cac78c1-0_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_372_610_raw3dgs/2013_05_28_drive_0000_sync_0000000372_0000000610/350d9a39-f_test"


# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2501_2706/2013_05_28_drive_0000_sync_0000002501_0000002706/69aa0444-f_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2501_2706_colmap/2013_05_28_drive_0000_sync_0000002501_0000002706/bbc015ea-b_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2501_2706_lidar/2013_05_28_drive_0000_sync_0000002501_0000002706/89cc6eac-9_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2501_2706_raw3dgs/2013_05_28_drive_0000_sync_0000002501_0000002706/a4c4b0ad-1_test"


# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2913_3233/2013_05_28_drive_0000_sync_0000002913_0000003233/a1901bba-8_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2913_3233_lidar/2013_05_28_drive_0000_sync_0000002913_0000003233/f0c86a4b-0_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2913_3233_colmap/2013_05_28_drive_0000_sync_0000002913_0000003233/3d585038-7_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2913_3233_raw3dgs/2013_05_28_drive_0000_sync_0000002913_0000003233/d853ef1c-9_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2913_3233_lpips_0.5/2013_05_28_drive_0000_sync_0000002913_0000003233/7921b4ed-3_test"


# "/workspace/WorkSpaceRec/vegs/bash_scripts/output/2013_05_28_drive_0009_sync_0000003972_0000004258/180f58e4-8_test" 
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_3972_4258_colmap/2013_05_28_drive_0009_sync_0000003972_0000004258/2c9625fb-d_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_3972_4258_lidar/2013_05_28_drive_0009_sync_0000003972_0000004258/288ba927-a_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_3972_4258_raw3dgs/2013_05_28_drive_0009_sync_0000003972_0000004258/04b55508-c_test"


# Define an array of strings
str_list=(
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2501_2706_colmap/2013_05_28_drive_0000_sync_0000002501_0000002706/bbc015ea-b_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2501_2706_lidar/2013_05_28_drive_0000_sync_0000002501_0000002706/89cc6eac-9_test"
# "/workspace/WorkSpaceRec/vegs/bash_scripts/output_2501_2706_raw3dgs/2013_05_28_drive_0000_sync_0000002501_0000002706/a4c4b0ad-1_test"
)

# Iterate over each string and call the Python script
for str in "${str_list[@]}"
do
  python ../render_video.py -m ${str} --data_type kitti360 --source_path /public/KITTI360
done



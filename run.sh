#!/bin/bash

cd /app
source "/opt/ros/$ROS_DISTRO/setup.bash"

MODEL=tartanvo_1914.pkl
DATASET_DIR=/datasets/KITTI/KITTI_10/image_left
RVIZ_CONFIG=./config.rviz

# Enable tracing
set -x

while getopts "ro" opt
do
  case $opt in
    # VAR=$OPTARG
    r)
        python3 vo_trajectory_from_folder.py \
            --model-name ${MODEL} \
            --kitti \
            --batch-size 1 --worker-num 1 \
            --test-dir ${DATASET_DIR}
        ;;
    o)
        rosparam set /img_dir ${DATASET_DIR}
        python3 tartanvo_node.py \
        & rosrun rviz rviz -d ${RVIZ_CONFIG}
        ;;
    \?) 
        echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    *)
        echo "*"
        ;;
  esac
done

# Disable tracing
set +x

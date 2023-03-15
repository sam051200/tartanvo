#!/bin/bash

cd /app
source "/opt/ros/$ROS_DISTRO/setup.bash"

MODEL=tartanvo_1914.pkl
RVIZ_CONFIG=./config.rviz

# Enable tracing
set -x

GT_DIR=""
while getopts "ked:g:ro" opt
do
  case $opt in
    # VAR=$OPTARG
    k)
        DATASET_FORMAT=--kitti
        ;;
    e)
        DATASET_FORMAT=--euroc
        ;;
    d)
        DATASET_DIR=$OPTARG
        ;;
    g)  
        GT_DIR="--pose-file ${OPTARG}"
        ;;
    r)
        python3 vo_trajectory_from_folder.py \
            --model-name ${MODEL} \
            ${DATASET_FORMAT} \
            --batch-size 1 --worker-num 1 \
            --test-dir ${DATASET_DIR} ${GT_DIR}
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

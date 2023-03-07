cd /app
python3 vo_trajectory_from_folder.py \
    --model-name tartanvo_1914.pkl \
    --kitti \
    --batch-size 1 --worker-num 1 \
    --test-dir /datasets/KITTI/KITTI_10/image_left

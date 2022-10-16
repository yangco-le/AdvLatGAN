CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./data --result_dir './stl_vanilla' --dataset 'stl10' &
CUDA_VISIBLE_DEVICES=3 python train.py --dataroot ./data --result_dir './stl_div' --div --dataset 'stl10' &
wait
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./data --result_dir './cifar_vanilla' &
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./data --result_dir './cifar_div' --div &
wait
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot ./data --resume ./stl_vanilla/trial/00199.pth --result_dir ./stl_vanilla --dataset 'stl10' &
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./data --resume ./stl_div/trial/00199.pth --result_dir ./stl_div --dataset 'stl10' &
wait

cd evaluate
bash genlog_stl.sh ../stl_vanilla/CIFAR10 ../stl_vanilla/CIFAR10
bash eval_stl.sh ../stl_vanilla/CIFAR10  ../stl_real  0 >../stl_vanilla/result.txt
cat ../stl_vanilla/result.txt
cd ..

cd evaluate
bash genlog_stl.sh ../stl_div/CIFAR10 ../stl_div/CIFAR10
bash eval_stl.sh ../stl_div/CIFAR10  ../stl_real  0 >../stl_div/result.txt
cat ../stl_div/result.txt
cd ..


#CUDA_VISIBLE_DEVICES=0 python test.py --dataroot ./data --resume ./cifar_vanilla/trial/00199.pth --result_dir ./cifar_vanilla &
#CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./data --resume ./cifar_div/trial/00199.pth --result_dir ./cifar_div &
#wait

cd evaluate
#bash genlog_cifar.sh ../cifar_vanilla/CIFAR10 ../cifar_vanilla/CIFAR10
bash eval_cifar.sh ../cifar_vanilla/CIFAR10  ../cifar_real  0 >../cifar_vanilla/result.txt
#cat ../cifar_vanilla/result.txt
cd ..

#cd evaluate
#bash genlog_cifar.sh ../cifar_div/CIFAR10 ../cifar_div/CIFAR10
#bash eval_cifar.sh ../cifar_div/CIFAR10  ../cifar_real  0 >../cifar_div/result.txt
#cat ../cifar_div/result.txt
#cd ..
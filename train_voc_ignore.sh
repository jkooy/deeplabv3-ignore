CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ignore.py --backbone resnet --lr 0.007 --workers 4 --use-sbd --loss-type pw --epochs 50 --batch-size 8 --gpu-ids 0,1,2,3 --checkname deeplab-resnet-ignore-2 --eval-interval 1 --dataset pascal

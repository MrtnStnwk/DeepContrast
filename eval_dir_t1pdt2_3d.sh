#/bin/bash
module load cuda/10.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/10.1/targets/x86_64-linux/lib/

S=128
CUDA_VISIBLE_DEVICES=1 python3 deepcontrast.py --image_folder=dir_t1pdt2_nii --ndims=3 --batch_size=2 --network=isensee5-1 --nfilters=32 --normalization=instance --shapeAi=${S} --shapeAj=${S} --shapeAk=${S} --channelsA=1 --shapeBi=${S} --shapeBj=${S} --shapeBk=${S} --channelsB=3 --loss_G=mae --trainmode=patch --mapping=regression --lr_G=0.0002 --workers=12 --pre_norm --augment_spatial --use_gan 

exit

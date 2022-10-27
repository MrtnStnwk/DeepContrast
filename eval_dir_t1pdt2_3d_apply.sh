#/bin/bash
module load cuda/10.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/10.1/targets/x86_64-linux/lib/

S=128
CUDA_VISIBLE_DEVICES="" python3 deepcontrast.py --image_folder=dir_t1pdt2_nii --ndims=3 --batch_size=2 --network=isensee5-1 --nfilters=32 --normalization=instance --shapeAi=${S} --shapeAj=${S} --shapeAk=${S} --channelsA=1 --shapeBi=${S} --shapeBj=${S} --shapeBk=${S} --channelsB=3 --loss_G=mae --trainmode=patch --lr_G=0.0002 --workers=12 --multi_gpu=1 --pre_norm --augment_spatial --use_gan --initG=saved_models/20200308-100636_pix2pix/model_1_weights_epoch_350 --applyFCN 

exit

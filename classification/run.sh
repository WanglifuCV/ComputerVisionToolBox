python train.py --backbone vgg19 \
                --gpu_list 1 \
                --regularizer 0.005 \
                --learning_rate 1e-3 \
                --dataset_name cifar100 \
                --steps_per_epoch 1000 \
                --architecture cifar \
                --epochs 2000 \
                --batch_size 256 \
                --batch_norm_type before_activation \
                --model_save_dir /home/wanglifu/learning/Deep-Learning-with-Python/ComputerVisionToolBox/classification/models/cifar100/vgg_256_channel_regularizer_l2_0.005_global_pooling \
                --category_num 100 \
                --input_size 32 \
                --final_pooling global_average_pooling \
                --lr_decay 0.95
                
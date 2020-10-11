python train.py --backbone resnet50 \
                --gpu_list 3 \
                --steps_per_epoch 500 \
                --epochs 50 \
                --batch_size 16 \
                --model_save_dir /home/lifu/learning/keras/ComputerVisionToolBox/classification/models/cats-vs-dogs/resnet50_keras_application/ \
                --category_num 2 \
                --input_size 224
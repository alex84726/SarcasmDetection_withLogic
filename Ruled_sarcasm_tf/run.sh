python3 train.py  --num_epochs 100 \
                  --data_file ../Data/train_balanced.npy \
                  --filter_sizes 3,4,5\
                  --num_filters 128 \
                  --dropout_keep_prob 0.5 \
                  # --l2_reg_lambda 0.3 \
                  --gpu_usage 0.2

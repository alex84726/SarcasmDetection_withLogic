python3 train_logic.py  --num_epochs 20 \
                  --data_file ../Data/train_balanced.npy \
                  --fea_file ../Data/train_balanced.fea.npy \
                  --filter_sizes 3,4,5\
                  --num_filters 128 \
                  --dropout_keep_prob 0.7 \
                  --l2_reg_lambda 0.1 \
                  --embedding_dim 300 \
                  --word2vec True \
                  --gpu_usage 1.0 \
                  --checkpoint_every 200\
                  --pi_curve exp_arise \
                  --pi_params 0.7,1e-5 \
                  --train_word2vec True

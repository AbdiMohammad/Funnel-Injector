python prune_timm_vit.py \
    --model_name vit_base_patch16_224 \
    --pruning_type l1 \
    --pruning_ratio 0.5 \
    --taylor_batchs 10 \
    --data_path data/imagenet \
    --test_accuracy \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --save_as output/pruned/vit_base_patch16_224_pruned_l1_uniform.pth \
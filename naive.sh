
mkdir -p log


for dir in "bonsai" "bicycle" "counter" "garden" "kitchen" "room" "stump"
do
# python train.py -s ./data/$dir/ --grad_weight 0.7 --chroma_edge_weight 1 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted.log

# python train.py -s ./data/$dir/ --grad_weight 1.0 --chroma_edge_weight 1.2 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted_2.log

# python train.py -s ./data/$dir/ --grad_weight 1.2 --chroma_edge_weight 1.5 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted_3.log



python train.py \
  -s ./data/$dir/ --eval \
  --expname "gray2rgb_SH_op_${dir}" \
  --two_stage --rgb_finetune_iters 100 \
  --stage2_feature_lr_scale 0.1 \
  --stage2_train_opacity --stage2_opacity_lr_scale 10 \
  --stage2_warmup_iters 50 \
  | tee -a log/${dir}_gray2rgb_stage2_DC_100_op.log

python train.py \
  -s ./data/$dir/ --eval \
  --expname "gray2rgb_SH_op_${dir}" \
  --two_stage --rgb_finetune_iters 50 \
  --stage2_feature_lr_scale 0.1 \
  --stage2_train_opacity --stage2_opacity_lr_scale 10 \
  --stage2_warmup_iters 30 \
  | tee -a log/${dir}_gray2rgb_stage2_DC_30_op.log

done



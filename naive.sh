
mkdir -p log


for dir in "bonsai" "bicycle" "counter" "garden" "kitchen" "room" "stump"
do
# python train.py -s ./data/$dir/ --grad_weight 0.7 --chroma_edge_weight 1 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted.log

# python train.py -s ./data/$dir/ --grad_weight 1.0 --chroma_edge_weight 1.2 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted_2.log

# python train.py -s ./data/$dir/ --grad_weight 1.2 --chroma_edge_weight 1.5 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted_3.log



# python train.py -s ./data/$dir/ --eval
#   --save_test_preds \
#   --force_DC_SH \
#   --expname "rgb_baseline_${dir}" | tee -a log/${dir}_rgb_DC_baseline.log

python train.py -s ./data/$dir/ --eval \
  --expname "_stage2_DC_rgb_500${dir}" \
  --save_test_preds \
  --force_DC_SH \
  --stage2_train_opacity \
  --stage2_feature_lr_scale 10 \
  --two_stage --rgb_finetune_iters 500 \
  --stage2_feature_lr_scale 10 --stage2_warmup_iters 200 | tee -a log/${dir}_stage2_DC_rgb_500.log


python train.py -s ./data/$dir/ --eval \
  --expname "_stage2_DC_rgb_200${dir}" \
  --save_test_preds \
  --force_DC_SH \
  --stage2_train_opacity \
  --stage2_feature_lr_scale 10 \
  --two_stage --rgb_finetune_iters 200 \
  --stage2_feature_lr_scale 10 --stage2_warmup_iters 100 | tee -a log/${dir}_stage2_DC_rgb_200.log

python train.py -s ./data/$dir/ --eval \
  --expname "_stage2_DC_rgb_1000${dir}" \
  --save_test_preds \
  --force_DC_SH \
  --stage2_train_opacity \
  --stage2_feature_lr_scale 10 \
  --two_stage --rgb_finetune_iters 1000 \
  --stage2_feature_lr_scale 10 --stage2_warmup_iters 500 | tee -a log/${dir}_stage2_DC_rgb_1000.log


done
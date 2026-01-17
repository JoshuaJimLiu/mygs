
mkdir -p log


for dir in "bonsai" "bicycle" "counter" "garden" "kitchen" "room" "stump"
do
# python train.py -s ./data/$dir/ --grad_weight 0.7 --chroma_edge_weight 1 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted.log

# python train.py -s ./data/$dir/ --grad_weight 1.0 --chroma_edge_weight 1.2 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted_2.log

# python train.py -s ./data/$dir/ --grad_weight 1.2 --chroma_edge_weight 1.5 --edge_quantile 0.995 --eval | tee -a log/${dir}_gray_usesh_weighted_3.log


python train.py -s ./dataGray/$dir/ --eval --color_loss "rgb" | tee -a log/${dir}_gray_as_rgb_no_sh.log

done



mkdir -p log


for dir in "bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump"
do
python train.py -s ./data/$dir/ --eval | tee -a log/$dir.log

python train.py -s ./data/$dir/ --eval --color_loss "rgb" | tee -a log/$dir_rgb.log 

done


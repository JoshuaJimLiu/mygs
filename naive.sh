
mkdir -p log

python train.py -s ./data/bicycle/ | tee -a log/bicycle.log

python train.py -s ./data/bonsai/  | tee -a log/bonsai.log

python train.py -s ./data/bicycle/ --color_loss "rgb" | tee -a log/bicycle_rgb.log

python train.py -s ./data/bonsai/ --color_loss "rgb" | tee -a log/bonsai_rgb.log

mkdir -p log

python train.py -s ./data/bicycle/ | tee -a log/bicycle.log

python train.py -s ./data/bonsai/  | tee -a log/bonsai.log
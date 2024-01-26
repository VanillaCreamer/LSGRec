python trainer.py --dataset yelp --version 1 --aggregate pandgnn --K 40 --lr 5e-3 --num_layer 2 --dim 64 --reg 5e-5 --epoch 201
wait
python trainer.py --dataset amazon --version 1 --aggregate pandgnn --K 40 --lr 1e-2 --num_layer 2 --dim 64 --reg 1e-5 --epoch 201
wait
python trainer.py --dataset beauty --version 1 --aggregate pandgnn --K 40 --lr 3e-3 --num_layer 2 --dim 64 --reg 2e-5 --epoch 201
wait
python .\converter.py --dataset compas --sensitive race --embedding_dim 6
python .\converter.py --dataset adult --sensitive sex --embedding_dim 10
python .\converter.py --dataset german --sensitive A13 --embedding_dim 16


python .\dualfair.py --dataset compas --sensitive race --num_workers 6 --embedding_dim 6
python .\dualfair.py --dataset adult --sensitive sex --num_workers 6 --embedding_dim 10
python .\dualfair.py --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16


rem replace save_pre with the real save_pre
python .\evaluate.py --save_pre dualfair_german_A13_seed_17_0504-20-30 --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16
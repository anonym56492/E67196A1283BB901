@echo off

echo Seed: %1

python .\dualfair.py --dataset compas --sensitive race --num_workers 6 --embedding_dim 6 --seed %1
python .\dualfair.py --dataset adult --sensitive sex --num_workers 6 --embedding_dim 10 --seed %1
python .\dualfair.py --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16 --seed %1


REM python .\dualfair.py --dataset compas --sensitive race --num_workers 6 --embedding_dim 6 --seed 0
REM python .\dualfair.py --dataset adult --sensitive sex --num_workers 6 --embedding_dim 10 --seed 0
REM python .\dualfair.py --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16 --seed 0

REM python .\dualfair.py --dataset compas --sensitive race --num_workers 6 --embedding_dim 6 --seed 1
REM python .\dualfair.py --dataset adult --sensitive sex --num_workers 6 --embedding_dim 10 --seed 1
REM python .\dualfair.py --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16 --seed 1

REM python .\dualfair.py --dataset compas --sensitive race --num_workers 6 --embedding_dim 6 --seed 2
REM python .\dualfair.py --dataset adult --sensitive sex --num_workers 6 --embedding_dim 10 --seed 2
REM python .\dualfair.py --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16 --seed 2

REM python .\dualfair.py --dataset compas --sensitive race --num_workers 6 --embedding_dim 6 --seed 3
REM python .\dualfair.py --dataset adult --sensitive sex --num_workers 6 --embedding_dim 10 --seed 3
REM python .\dualfair.py --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16 --seed 3

REM python .\dualfair.py --dataset compas --sensitive race --num_workers 6 --embedding_dim 6 --seed 4
REM python .\dualfair.py --dataset adult --sensitive sex --num_workers 6 --embedding_dim 10 --seed 4
REM python .\dualfair.py --dataset german --sensitive A13 --num_workers 6 --embedding_dim 16 --seed 4
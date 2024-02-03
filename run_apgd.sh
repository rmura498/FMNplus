#!/bin/bash

python run_apgd_baseline.py --model_id 8 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 8 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 7 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 7 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 6 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 6 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 5 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 5 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 4 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 4 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 3 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 3 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 2 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 2 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 1 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 1 --batch_size 125 --n_batches 8 --steps 1000 --loss CE

python run_apgd_baseline.py --model_id 0 --batch_size 125 --n_batches 8 --steps 1000 --loss DLR
python run_apgd_baseline.py --model_id 0 --batch_size 125 --n_batches 8 --steps 1000 --loss CE
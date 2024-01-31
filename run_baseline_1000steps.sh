#!/bin/bash

python run_fmn_baseline.py --model_id 0 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 1 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 2 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 3 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 4 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 5 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 6 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 7 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5
python run_fmn_baseline.py --model_id 8 --optimizer SGD --scheduler CALR --loss LL --steps 1000 --device cuda --cuda_device 1 --batch_size 200 --n_batch 5


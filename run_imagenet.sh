#!/bin/bash

python ax_tune.py --model_id 15 --batch_size 64 --optimizer SGD --scheduler CALR --steps 300 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 15 --batch_size 64 --optimizer SGD --scheduler CALR --steps 300 --loss LL --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 15 --batch_size 64 --optimizer Adam --scheduler None --steps 300 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 15 --batch_size 64 --optimizer SGD --scheduler RLROPVec --steps 300 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 15 --batch_size 64 --optimizer Adamax --scheduler None --steps 300 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle

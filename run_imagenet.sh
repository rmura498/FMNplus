#!/bin/bash

python ax_tune.py --model_id 14 --batch_size 32 --optimizer SGD --scheduler CALR --steps 200 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 14 --batch_size 32 --optimizer SGD --scheduler CALR --steps 200 --loss LL --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 14 --batch_size 32 --optimizer Adam --scheduler None --steps 200 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 14 --batch_size 32 --optimizer SGD --scheduler RLROPVec --steps 200 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle
python ax_tune.py --model_id 14 --batch_size 32 --optimizer Adamax --scheduler None --steps 200 --loss DLR --gradient_update Sign --n_trials 32 --device cuda --cuda_device 0 --shuffle

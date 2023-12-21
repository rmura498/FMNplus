#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <steps> <batch_size> <batches> <cuda_device> <model_id> <alpha_init>"
    exit 1
fi

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer Adam --scheduler None --loss DLR --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer Adam --scheduler None --loss CE --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer Adam --scheduler None --loss LL --shuffle False --alpha_init $6 

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer Adamax --scheduler None --loss DLR --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer Adamax --scheduler None --loss CE --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer Adamax --scheduler None --loss LL --shuffle False --alpha_init $6 

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer SGD --scheduler CALR --loss DLR --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer SGD --scheduler CALR --loss CE --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNBase --optimizer SGD --scheduler CALR --loss LL --shuffle False --alpha_init $6 

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNVec --optimizer SGD --scheduler RLROPVec --loss DLR --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNVec --optimizer SGD --scheduler RLROPVec --loss CE --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type FMNVec --optimizer SGD --scheduler RLROPVec --loss LL --shuffle False --alpha_init $6 

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type AA --loss CE --shuffle False --alpha_init $6 &
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --force_cpu True\
            --attack_type AA --loss DLR --shuffle False --alpha_init $6 

         
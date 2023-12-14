#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <steps> <batch_size> <batches> <cuda_device> <model_id> <alpha_init>"
    exit 1
fi

echo "### Running FMN base x6 Versions ... ###"
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer SGD --scheduler CALR --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer SGD --scheduler CALR --loss DLR --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer Adam --scheduler RLROP --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer Adam --scheduler RLROP --loss DLR --shuffle False --alpha_init $6

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer SGD --scheduler CALR --loss LL --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer Adam --scheduler RLROP --loss LL --shuffle False --alpha_init $6

echo "### Running FMN vec RLROPVec x8 Versions ... ###"
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer SGD --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer SGD --loss DLR --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer Adam --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer Adam --loss DLR --shuffle False --alpha_init $6
            
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer SGD --loss LL --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer Adam --loss LL --shuffle False --alpha_init $6

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer Adam --loss CE --shuffle False --alpha_init $6 --extra_iters True
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler RLROPVec --optimizer Adam --loss DLR --shuffle False --alpha_init $6 --extra_iters True

echo "### Running FMN vec x8 CALRVec Versions ... ###"
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer SGD --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer SGD --loss DLR --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer Adam --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer Adam --loss DLR --shuffle False --alpha_init $6

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer SGD --loss LL --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer Adam --loss LL --shuffle False --alpha_init $6

python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer Adam --loss CE --shuffle False --alpha_init $6 --extra_iters True
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNVec --scheduler CALRVec --optimizer Adam --loss DLR --shuffle False --alpha_init $6 --extra_iters True

echo "### Running FMN base (no scheduler) x4 Versions ... ###"
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer SGD --scheduler None --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer SGD --scheduler None --loss DLR --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer Adam --scheduler None --loss CE --shuffle False --alpha_init $6
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type FMNBase --optimizer Adam --scheduler None --loss DLR --shuffle False --alpha_init $6

echo "### Running AA x2 Versions ... ###"
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type AA --loss CE --shuffle False
python test_v_scheduler_2.py --model_id $5 --steps $1 --batch_size $2 --num_batches $3 --cuda_device $4 \
            --attack_type AA --loss DLR --shuffle False


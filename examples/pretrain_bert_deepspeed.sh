#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=<Specify path and file prefix>_text_sentence
CHECKPOINT_PATH=<Specify path>

config_json=deepspeed_config.json

#ZeRO Configs
stage=0
rbs=50000000
agbs=5000000000


deepspeed --num_nodes ${NNODES} --num_gpus ${GPUS_PER_NODE} pretrain_bert.py \
       --deepspeed \
       --zero-reduce-scatter \
       --zero-contigious-gradients \
       --deepspeed_config ${config_json} \
       --zero-stage ${stage} \
       --zero-reduce-bucket-size ${rbs} \
       --zero-allgather-bucket-size ${agbs} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file bert-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16

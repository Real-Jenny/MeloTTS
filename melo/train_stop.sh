#!/bin/bash

CONFIG=$1
GPUS=$2
MODEL_NAME=$(basename "$(dirname $CONFIG)")

PORT=10902

while : 
do
    # 运行训练，并捕获退出状态码
    torchrun --nproc_per_node=$GPUS \
        --master_port=$PORT \
        train.py --c $CONFIG --model $MODEL_NAME
    
    EXIT_CODE=$?
    
    # 如果退出码为0（正常完成），则跳出循环
    if [ $EXIT_CODE -eq 0 ]; then
        echo "训练正常完成，退出。"
        break
    else
        echo "训练异常退出（退出码: $EXIT_CODE），30秒后重启..."
        # 清理残留进程
        for PID in $(ps -aux | grep $CONFIG | grep python | awk '{print $2}')
        do
            echo "清理进程: $PID"
            kill -9 $PID 2>/dev/null
        done
        sleep 30
    fi
done

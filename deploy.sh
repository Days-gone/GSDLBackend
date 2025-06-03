#!/bin/bash

IMAGE_NAME="gsdlbackend"
CONTAINER_NAME="gsdlbackend-container"

echo "停止并删除旧容器..."
sudo docker stop $CONTAINER_NAME 2>/dev/null || true
sudo docker rm $CONTAINER_NAME 2>/dev/null || true

echo "正在删除旧镜像..."
sudo docker rmi $IMAGE_NAME 2>/dev/null || true

echo "构建新的DL-Docker 镜像..."
sudo docker build -t $IMAGE_NAME .

# 检查构建是否成功
if [ $? -eq 0 ]; then
    echo "镜像构建成功！正在启动容器..."
    
    # 启动新容器，使用主机网络，交互模式以显示终端输出
    sudo docker run -it \
        --name $CONTAINER_NAME \
        --network host \
        $IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        echo "容器启动成功！"
        echo "应用正在运行在 http://localhost:8001"
        echo "容器名称: $CONTAINER_NAME"
    else
        echo "容器启动失败！"
        exit 1
    fi
else
    echo "镜像构建失败！"
    exit 1
fi
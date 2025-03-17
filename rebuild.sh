sudo docker rm inst_dlserver

sudo docker build -t dlserver .

sudo docker run --name inst_dlserver --network host dlserver

# 如果容器已经存在，可以使用以下命令启动容器
# sudo docker start --network host inst_dlserver
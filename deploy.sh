aws ecr get-login-password --region sa-east-1 | sudo docker login --username AWS --password-stdin 689566614228.dkr.ecr.sa-east-1.amazonaws.com
sudo docker build -t defend .
sudo docker tag defend:latest 689566614228.dkr.ecr.sa-east-1.amazonaws.com/defend:latest
sudo docker push 689566614228.dkr.ecr.sa-east-1.amazonaws.com/defend:latest

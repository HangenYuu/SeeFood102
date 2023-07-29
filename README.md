# SeeFood102

## Status
- ✅ Fine-tune the model on multiple GPUs with PyTorch Lightning
- ✅ Monitor finetuning with Tensorboard and upload the runs online
- ✅ Store the weights in an Amazon S3 bucket
- ✅ Version track the model weights with DVC
- ✅ Convert the PyTorch Lightning checkpoint to ONNX weights and use ONNX Runtime for inference
- ✅ Create the backend with FastAPI and a minimal HTML/Javascript frontend, allowing users to upload image and receive top-5 categories + probabilities
- ✅ Containerize the backend with Docker
- ✅ Store the Docker image in AWS Elastic Container Registry (ECR)
- ✅ Create a GitHub action to automatically construct the Docker image and push to ECR on pushing code to the GitHub repository.
- ❌ Deployment - currently encountering issues. Intended options:
  - ❌ Deploy the Docker image on AWS Elastic Container Service (ECS).
    - Reason: ECS requires a cluster, task, and service. The cluster was created successfully, but the service creation was failed repeatedly.
    - Possible remedy 1: Switch to an EC2 instance (straightforward, but not scalable)
    - Possible remedy 2: Switch to a Lambda function (does not allow for environment specification nor fastAPI easyness since I have to use Lambda function format)
    - Possible remedy 3: Switch to a different vendor (Digital Ocean, GCP, etc.)
- [ ] Add to the GitHub action to automatically deploy the Docker image on after construction
- [ ] Setup monitoring with AWS CloudWatch logs, Elastic Search Cluster (to stream the logs), and a dashboard service (Kibana, Grafana, EvidentlyAI, etc.)

## Setup environment
```sh
conda create -n seefood102 python=3.10
conda activate seefood102
pip install -r requirements.txt
```

## Model training
I chose a pretrained LeVit ([paper](https://arxiv.org/pdf/2104.01136.pdf), [code](https://github.com/facebookresearch/LeViT), [weight](https://huggingface.co/timm/levit_256.fb_dist_in1k)) and then fine-tuned with all parameters unfrozen on the Food101 Dataset. The training code was written with PyTorch Lightning to reduce the amount of boilerplate and utilize the 2 GPUs I was given. I used Tensorboard to monitor the training. The training can be viewed [here](https://tensorboard.dev/experiment/gX8buBf7TJOW8RytJaCA7g/#scalars). At the end, the model achieved ~0.78 F1 and accuracy.

## Setup DVC
I used DVC (Data Version Control), a Git-like tool for data, to track the model weight. The setup depends on your choice of cloud storage. I ~~was forced to choose S3 to increase the prospect of getting a job~~ chose AWS S3 for its popularity and security despite the initial effort needed to set things one bucket (which actually was easy compared to other things AWS offers - more on that later). In any case, from [the documentation](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3#custom-authentication), the steps as of July 2023 are:

1. Go to **IAM Management Console**.
2. Create a new user (recommended) with specified permissions. Besides AWS S3, I want to use this User for the task of ECR and ECS, so I also give the User permissions for these services as needed.
3. I chose to use the option of access key ID/secret with no multi-factor authentication (MFA) token. That seems to be the most convenient choice as it also applies to other services in the GitHub Action later.

## 

*Pizza image by Pablo Pacheco via [Unsplash](https://unsplash.com/photos/D3Mag4BKqns)*

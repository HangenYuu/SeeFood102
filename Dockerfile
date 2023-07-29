FROM continuumio/miniconda3:23.3.1-0

COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# install requirements
RUN pip install awslambdaric
RUN pip install "dvc[s3]"
RUN pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements_inference.txt

# initialize dvc
RUN dvc init --no-scm
# connect to remote server
RUN dvc remote add -d awsremote s3://food101
RUN dvc remote modify awsremote version_aware true
RUN dvc config core.analytics false
RUN dvc remote modify --local awsremote access_key_id $AWS_ACCESS_KEY_ID
RUN dvc remote modify --local awsremote secret_access_key $AWS_SECRET_ACCESS_KEY

RUN cat .dvc/config

RUN dvc pull models/levit_256/onnx/checkpoints.onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD [ "lambda_handler.lambda_handler"]
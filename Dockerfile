FROM amazon/aws-lambda-python:3.11

COPY ./ /app
WORKDIR /app

ARG MODEL_DIR=./models
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

# install requirements
RUN yum install git gcc-c++ -y
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

ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR
CMD [ "lambda_handler.lambda_handler"]
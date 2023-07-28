FROM amazon/aws-lambda-python:3.10.2023.07.19.04


ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

# install requirements
RUN yum install -y git gcc-c++ wget tar gcc
RUN wget https://sqlite.org/2023/sqlite-autoconf-3420000.tar.gz
RUN tar xvfz sqlite-autoconf-3350500.tar.gz
RUN cd sqlite-autoconf-3350500 && ./configure --prefix=/usr && make && make install
COPY requirements_inference.txt requirements_inference.txt
RUN pip install "dvc[s3]"
RUN pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install -r requirements_inference.txt --no-cache-dir
COPY ./ ./
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

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

RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR
EXPOSE 8000
CMD ["lambda_handler.lambda_handler"]
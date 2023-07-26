FROM continuumio/miniconda3:23.3.1-0
COPY ./ /app
WORKDIR /app
RUN pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements_inference.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
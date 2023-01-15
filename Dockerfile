FROM ubuntu:latest
MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py 
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
#    pip install numpy scipy==1.1.0 scikit-learn==0.22.2 pandas numpy datetime re math os pickle warnings time pyarrow boto3 logging nltk xlrd flask gevent gunicorn 

RUN rm -rf /root/.cache


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY inference /opt/program
WORKDIR /opt/program
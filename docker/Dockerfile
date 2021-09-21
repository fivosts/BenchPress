FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y sudo git wget

RUN git clone https://github.com/jordiae/clgen.git

WORKDIR /home/clgen
RUN echo "cmake y" | bash requirements.apt

WORKDIR /home/clgen
RUN mkdir build
WORKDIR /home/clgen/build
RUN cmake ..
RUN make -j 12

# Download C
CMD ./clgen --config model_zoo/github/bq_C_db.pbtxt

FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y sudo git wget

RUN git clone https://54b082ffbdcd8ad60dcba3ba992863b40069667f@github.com/fivosts/benchpress.git

WORKDIR /home/benchpress
RUN echo "cmake y" | bash requirements.apt

WORKDIR /home/benchpress
RUN mkdir build
WORKDIR /home/benchpress/build
RUN cmake ..
RUN make -j 12

# Run a simple example
CMD ./clgen --min_samples 10 --config model_zoo/BERT/tiny.pbtxt

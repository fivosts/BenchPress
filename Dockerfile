FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y sudo git

RUN git clone https://3f2defbceff83ef75197a0d924fd2d96ef86e327@github.com/fivosts/clgen.git

WORKDIR /home/clgen
RUN bash requirements.apt

WORKDIR /home/
RUN wget https://cmake.org/files/v3.13/cmake-3.13.4.tar.gz
RUN tar -xvf cmake-3.13.4.tar.gz
WORKDIR /home/cmake-3.13.4
RUN ./bootstrap --prefix=/usr
RUN make
RUN make install
RUN cmake --version

WORKDIR /home/clgen
RUN mkdir build
WORKDIR /home/clgen/build
RUN cmake ..
RUN make -j 12

# Run a simple example
CMD ./clgen --min_samples 10 --config model_zoo/BERT/tiny.pbtxt

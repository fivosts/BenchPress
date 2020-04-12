FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y git python3.6 python3.7 python3-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.6 get-pip.py
RUN python3.7 get-pip.py

RUN git clone https://3f2defbceff83ef75197a0d924fd2d96ef86e327@github.com/fivosts/clgen.git

WORKDIR /home/clgen
RUN bash apt_deps.sh
RUN bazel build //deeplearning/clgen

# Run a simple example
CMD bazel-bin/deeplearning/clgen/clgen --config $PWD/deeplearning/clgen/tests/data/tiny/config.pbtxt

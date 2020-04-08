FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y git python3.6 python3.7 python3-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.6 get-pip.py
RUN python3.7 get-pip.py
RUN git clone https://github.com/fivosts/clgen.git

WORKDIR /home/clgen
RUN bash apt_deps.sh
RUN bazel build //deeplearning/clgen

CMD bazel-bin/deeplearning/clgen/clgen --config $PWD/deeplearning/clgen/tests/data/tiny/config.pbtxt

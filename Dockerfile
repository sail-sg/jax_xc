FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$PATH

WORKDIR $HOME

# install base dependencies

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update \
    && apt-get install -y git curl wget gcc-9 g++-9 build-essential patchelf make libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev swig libhdf5-dev
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

#  install pyenv and pythons
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PYENV_ROOT /root/.pyenv
ENV PATH "$PATH:$PYENV_ROOT/bin"
RUN pyenv install 3.7-dev
RUN pyenv install 3.8-dev
RUN pyenv install 3.9-dev
RUN pyenv install 3.10-dev
# RUN pyenv install 3.11-dev

COPY requirements.txt /app/
WORKDIR /app

RUN eval "$(pyenv init -)" && pyenv global 3.7-dev; pip3 install -r requirements.txt;
RUN eval "$(pyenv init -)" && pyenv global 3.8-dev; pip3 install -r requirements.txt;
RUN eval "$(pyenv init -)" && pyenv global 3.9-dev; pip3 install -r requirements.txt;
RUN eval "$(pyenv init -)" && pyenv global 3.10-dev; pip3 install -r requirements.txt;
# RUN eval "$(pyenv init -)" && pyenv global 3.11-dev; pip3 install -r requirements.txt;

# install go from source
RUN wget https://golang.org/dl/go1.17.3.linux-amd64.tar.gz
RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.17.3.linux-amd64.tar.gz
RUN ln -sf /usr/local/go/bin/go /usr/bin/go

# install bazel
RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

# WORKDIR /app

# COPY . .

# compile and test release wheels

# RUN for i in 7 8 9 10 11; do make pypi-wheel BAZELOPT="--remote_cache=http://bazel-cache.sail:8080"; pip3 install wheelhouse/*cp3$i*.whl; rm dist/*.whl; make release-test; done

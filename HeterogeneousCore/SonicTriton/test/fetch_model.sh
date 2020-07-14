#!/bin/bash

# borrowed from https://github.com/NVIDIA/triton-inference-server/tree/master/docs/examples

TRITON_VERSION=$(scram tool info triton-inference-server | grep "Version : " | cut -d' ' -f3)

MODEL_DIR=../data/models/resnet50_netdef
mkdir -p $MODEL_DIR
cd $MODEL_DIR

curl -O -L https://github.com/NVIDIA/triton-inference-server/raw/v${TRITON_VERSION}/docs/examples/model_repository/resnet50_netdef/config.pbtxt
curl -O -L https://github.com/NVIDIA/triton-inference-server/raw/v${TRITON_VERSION}/docs/examples/model_repository/resnet50_netdef/resnet50_labels.txt

mkdir -p 1

curl -o 1/model.netdef http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/predict_net.pb
curl -o 1/init_model.netdef http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/init_net.pb

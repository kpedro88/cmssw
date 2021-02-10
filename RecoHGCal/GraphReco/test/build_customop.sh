#!/bin/bash

IMAGE=${USER}/hgcalml:20.09-tfpepr-py3

git clone https://github.com/cms-pepr/HGCalML.git

docker build -t ${IMAGE} -f Dockerfile.customop .

docker create --name hgcalml_artifacts ${IMAGE}

docker cp hgcalml_artifacts:/workspace/libpeprops.so ../data/models/hgcal_oc_reco/

docker rm -f hgcalml_artifacts

rm -rf HGCalML

#!/bin/bash

apt-get update

if ! command -v pip &> /dev/null
then
    apt-get install -y python3-pip
fi

pip install medmnist

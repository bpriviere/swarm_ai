#!/usr/bin/env bash
# Installs dependencies for the Swarm Project
# Might need to edit python_requirements to make sure the correct version of pytorch gets installed (defaults to CUDA pytorch (which will fail on non-CUDA systems)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# General stuff
echo 
echo "Installing the apt stuff"
echo "============================"

sudo apt install -y build-essential gcc make cmake python3-dev libyaml-cpp-dev ffmpeg
sudo apt install -y libeigen3-dev libyaml-cpp-dev
sudo apt install python3-pip -y

# Python Stuff
echo 
echo "Installing the pip3 stuff"
echo "============================"

pip3 install -r $DIR/python_requirements.txt

# Good luck with Gurobi... see main repo README.md


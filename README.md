# swarm_ai
https://github.com/bpriviere/swarm_ai.git

## Working with the code
Developed using Ubuntu 20.20 (at least that's what Wolfgang uses, and has worked for matt too)
The code as it currently stands should take under a minute to run.

## Dependencies
These instructions were developed on a virtual machine on Windows running Xubuntu 20.04
### Setting up (X)ubuntu 20.04
Install dependencies for VM Addons (reboot required)
```
sudo apt-get install build-essential gcc make perl dkms
```


### Python3
Requires python3.X (X>=6 should be ok)
```
sudo apt install python3-pip -y
pip3 install wheel
pip3 install numpy gym pandas matplotlib cvxpy
```
For CUDA support (nvidea VGA required)
```
pip3 install pytorch
```
otherwise install
```
pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Gurobi
Download from (you will need to make an account, but academic licence is free)
```
https://www.gurobi.com/downloads/gurobi-software/
```
Follow the instructions at (remember to fix the folder names)
```
https://www.gurobi.com/documentation/8.1/quickstart_linux/software_installation_guid.html
```
Get an academic licence key (grbgetkey 2d737662-c219-11ea-bd95-0a7c4f30bdbe)
```
https://www.gurobi.com/downloads/end-user-license-agreement-academic/
```
and activate the key using the tool (found in the installation directory /opt/gurobiXXX/linux64/bin/
```
./grbgetkey
```

## Running the software

### Regular MPC controller

Set `controller_name` to `controller/joint_mpc.py` in `param.py`.

```
code$ python3 run.py
```

### Learning

#### Generate data and Generate Policy

* In `glas/gparam.py` set `make_raw_data_on` and `make_labelled_data_on` to True in order to generate data.
* Set `train_model_on` in order to use learning.

```
code/glas$ python3 grun.py
```

#### Evaluation

Set `controller_name` to `controller/glas.py` in `param.py`.

```
code$ python3 run.py
```

## Visualizing result

```
python3 plotter.py ../current_results/sim_result_0.pickle --outputPDF test.pdf
python3 plotter.py ../current_results/sim_result_0.pickle --outputMP4 test.mp4
```

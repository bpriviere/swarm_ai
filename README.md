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

Install guest additions (Oracle VM Virtual Box)

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
Download from
```
https://www.gurobi.com/downloads/gurobi-software/
```
Follow the instructions at (remember to fix the folder names)
```
https://www.gurobi.com/documentation/8.1/quickstart_linux/software_installation_guid.html
```
Get an academic licence key.  You will need to make an account, but the academic licence is free.
```
https://www.gurobi.com/downloads/end-user-license-agreement-academic/
```
and activate the key using the tool (found in the installation directory /opt/gurobi902/linux64/bin/
```
./grbgetkey
```
Check that the install went correctly with 
```
gurobi_cl
```

When all else fails:
* Remember that installing gurobi sucks 
* Try running `sudo python3 /opt/gurobi902/linux64/setup.py`
* Put the environment variables into both `.bashrc` and `.profile`


### VS Code (IDE)
For those who want to use VS Code and forget how to install it
```
sudo snap install --classic code 
```
Upon opening a python file in VS Code, should be prompted to install a python add-on.
Install the python add-on and then make sure `Python 3.X.X 64-bit (/usr/bin/python3)` is selected
Also should install pylint (used for debugging) as well.

## Running the software
### Before First Run
Need to generate the data for using glas, or get a `/models/` directory and files from somewhere.

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

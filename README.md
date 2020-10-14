# swarm_ai
https://github.com/bpriviere/swarm_ai.git
```
git clone --recurse-submodules https://github.com/bpriviere/swarm_ai.git 
```

## Working with the code
Developed using Ubuntu 20.20 (at least that's what Wolfgang uses, and has worked for matt too)
The code as it currently stands should take under a minute to run.

## Dependencies
These instructions were developed on Xubuntu 20.04 and use python3.X (X>=6 should be ok). 

Additional dependencies for if running as a VM (reboot requried)
```
sudo apt-get install -y perl dkms
```

### Python3
Requires python3.X (X>=6 should be ok)
```
sudo apt install python3-pip -y
pip3 install wheel
pip3 install numpy gym pandas matplotlib opencv-python cvxpy PyPDF2
```
For CUDA support (nvidea VGA required)
>>>>>>> feature_curriculum
```
setup$ ./install_dependencies.sh
```
The `python_requirements.txt` file defaults to using CUDA support.  For systems that don't have CUDA, modify to ensure the correct things are installed.

### Installing Gurobi
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

### Compiling the cpp Components
Follow the instructions at 
```
https://github.com/bpriviere/swarm_ai/tree/master/code/cpp
```

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

For vis of training data, from `~/glas/`:
```
python grun.py -file ../../data/demonstration/labelled_atrain_0a_1b_0trial.npy
```

#### Evaluation

Set `controller_name` to `controller/glas.py` in `param.py`.

```
code$ python3 run.py
```

## Visualizing result
mp4 output requires ffmpeg to be installed.  This can be installed via apt (sudo apt install ffmpeg).  The script 'test_plotter.sh' will run these commands to make life easier

### Resources
Each team can be represented as an image (teamA.png, teamB.png).  The images are stored in `code/resources` and should be 300 x 300 pixels in size.

### Single Files
Single files can be converted by specifying the .pickle file to be used

```
code$ python3 plotter.py ../current_results/sim_result_0.pickle --outputPDF ../plots/test.pdf
code$ python3 plotter.py ../current_results/sim_result_0.pickle --outputMP4 ../plots/test.mp4
```

### Batch Processing
Batch processing can be invoked by specifying the folder (remember the trailing slash) to search for .pickle files.  This will search recursively so is good for checking each of the

```
code$ python3 plotter.py ../current_results/ --outputPDF ../plots/test.pdf --outputMP4 ../plots/test.mp4
```

## Implementing on RaspberryPi 4
Readme file for getting the code to execute on an embedded environment.  Tested on a RaspberryPi 4 (2GB)

### Setting up RaspberryPi 4
From the Ubuntu Website (https://ubuntu.com/download/raspberry-pi), and use the 64-bit version of Ubuntu 20.04.1
```
ubuntu-20.04.1-preinstalled-server-arm64+raspi.img
```
Follow the instruction at 
```
https://ubuntu.com/tutorials/how-to-install-ubuntu-on-your-raspberry-pi
```
to set the system up.  Don't forget to add a file `ssh` (no extension, empty file) to the boot directory to enable ssh into the system.  This step isn't mentioned in the walkthrough but ssh is disabled unless this file is present.

Other things during first log in
```
hostnamectl set-hostname swarm
sudo adduser swarm
sudo adduser swarm sudo
```

Restart RPi (`sudo reboot`), and log back in as `swarm`.  
```
sudo userdel -r ubuntu

sudo apt update
sudo apt upgrade -y
```

Install a desktop environment
```
sudo apt install -y xubuntu-desktop
sudo reboot
```

### Software Requirements
Follow as per above (modify for no CUDA support).  Download the repo
```
git clone --recurse-submodules https://github.com/bpriviere/swarm_ai.git 
```
and run the install script
```
setup$ ./install_dependencies.sh
```

### Running the code
Instructions at `https://github.com/bpriviere/swarm_ai/tree/master/code/cpp` but basically just run
```
code/cpp$ ./build_all.h
```
and where you run the test script
```
code/cpp$ python3 test_python_binding.py
```
just make sure to comment out the parts that need torch.


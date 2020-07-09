# swarm_ai

## Working with the code
Developed using Ubuntu 20.20 (at least that's what Wolfgang uses)

The code as it currently stands should take under a minute to run.

## Dependencies
### VS Code
Installed a standard version of VS Code (Windows x64)
Installed the C++ build tools (requried for building cvxpy)

### Python3
```
pip3 install wheel
pip3 install numpy, gym, pandas, matplotlib, cvxpy
```
Gurobi Python Interface
Download from (you may need an account)
```
https://www.gurobi.com/downloads/gurobi-software/
```
Run the installer
```
sudo python3 setup.py install
```
Get an academic licence key (grbgetkey 2d737662-c219-11ea-bd95-0a7c4f30bdbe)
```
https://www.gurobi.com/downloads/end-user-license-agreement-academic/
```
and activate the key using the tool
```
./bin/grbgetkey
```

Link Gurobi to the installed python.
Run 
```
C:\gurobi902\win64\bin\pysetup.bat
```
and use the local install (default is C:/Users/matt/AppData/Local/Programs/Python/Python38/ )

## Running the software
```
python3 run.py
```
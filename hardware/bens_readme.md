

## README CRAZYFLIE

objects: 
	- desktop 
	- laptop 
	- switch 
	- wand 


Preliminary 
- go to lab and clean space
- find box labelled crazyflie
- charge batteries (max 4 not 5 per charger, best are nanotech short cable, charged when light goes out)
- find crazyflies with label that corresponds to chooser.py 
- plug in lenovo [vikon] computer power, keyboard, mouse
	- connect orange ethernet from vikon desktop to vikon switch box 
	- connect gray ethernet (vikon computer to arcl_2g router yellow port)  
	- connect yellow ethernet (blue router port to wall)  
- turn on vikon software (Vicon Tracker 1.3.1) on desktop  
- turn on vikon hardware on switch  
- wait until cameras turn green 
- connect crayradio dongle extension (located by the switch) to laptop 
- connect xbox controller to laptop and have ros-noetic-joy installed 

Calibrate
- find wand 
- create camera mask (masks unwanted relfected objects... so hide wand and crazyflies)
	- in 'system' tab select all cameras 
	- in main tab (top right) select cameras so you see raw camera information 
	- in 'calibrate' tab on the top left hit 'start' for camera mask 
	- wait til everything visible turns blue 
	- hit stop 
- calibrate cameras
	- hit start for calibrate cameras in calibrate tab
	- use wand everywhere until the cameras all turn green, image error on the order of 0.25 (in pixels)
	- hit stop, wait for cameras to fold 
- set origin (0,0) point of space
	- (r,g,b) = (x,y,z), + y-axis is longest arm of wand 
	- hit start, select pivot point of wand and set origin 
	- check it in 3d perspective view 

Update Crazyflie firmware: 
- find a crazyflie without microSD card and plug in battery 
- run python chooser (see hardware readme) and select chosen crazyflie
- try battery check, reboot, and flash (STM) flash will take a minute 
- flash will send firmware from crazyswarm folder to crazyflie (firmware == whats runs onboard)

Flight Test: 
- put crazyflie visible to vikon with antenna (front) pointing in the +x dir
- follow other readme instructions (here are my paths)
- run roslaunch mice-ros-pkg lab.launch
- use xbox controller start to start and back to land, and 'b' to emergency kill

Adding a crazyflie: 
- add start location and radio and type in allCrazyflies.yaml
- connect radio with "cfclient" , connect, connect menu -> configure2.X -> change channel number to desired -> write and exit 
rerun chooser.py 
- remember to manually restart the drone! 
- add new robot to the run.launch file 

LEDs: 
- rings in 
- Check polarity at "https://www.bitcraze.io/documentation/tutorials/getting-started-with-expansion-decks/"
- turn flie off, put ring correctly, turn off 
- change 'effect' in launch.sim 

Cleanups: 
- close vicon software and turn off switch 
- dont turn off computer 
- put charged batteries in fireproof bag 
- clean 



Other READMES Ben specifics: 

- Terminal 1: Select/Deselect Crazyflies to Use (from "/home/ben/projects/swarm_ai/ros/crazyswarm/ros_ws/src/crazyswarm/scripts")
```
$ cd /home/ben/projects/swarm_ai/ros/crazyswarm/ros_ws/src/crazyswarm/scripts
$ source /home/ben/projects/swarm_ai/ros/crazyswarm/ros_ws/devel/setup.bash
$ python chooser2.py --configpath ../../userPackages/mice-ros-pkg/launch/
```
- Terminal 2: Run simulation or connect to robots (from "/home/ben/projects/swarm_ai/hardware/mice-ros-pkg/scripts")
	- Prepare terminal
```
$ cd /home/ben/projects/swarm_ai/hardware/mice-ros-pkg/scripts
$ source /home/ben/projects/swarm_ai/ros/crazyswarm/ros_ws/devel/setup.bash
$ export PYTHONPATH=$PYTHONPATH:/home/ben/projects/swarm_ai/ros/crazyswarm/ros_ws/src/crazyswarm/scripts
```
	- Run simulation only (robot position will be shown in rviz):
```
$ roslaunch mice-ros-pkg sim.launch
```
	- Run in lab (connection to selected crazyflies; robot position will be shown in rviz as obtained from motion capture):
```
$ roslaunch mice-ros-pkg lab.launch
```

- Terminal 3: Run simulation or connect to robots (from "/home/ben/projects/swarm_ai/hardware/mice-ros-pkg/scripts")
	- Prepare terminal
```
$ cd /home/ben/projects/swarm_ai/hardware/mice-ros-pkg/scripts
$ source /home/ben/projects/swarm_ai/ros/crazyswarm/ros_ws/devel/setup.bash
$ export PYTHONPATH=$PYTHONPATH:/home/ben/projects/swarm_ai/ros/crazyswarm/ros_ws/src/crazyswarm/scripts
```
	- Runs robot.py 
```
$ roslaunch mice-ros-pkg run.launch
```


: solutions run ok, but dont look taht intersting! 
Ideas: 
	- minimum speed 
	- bigger goal region 
	- enforce velocity and acceleration limits of crazyflie, etc. 
	- 
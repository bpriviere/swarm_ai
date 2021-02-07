# Hardware Experiments for Mice

Distributed implementation using the Crazyswarm framework. Each robot executes its own script (mice-ros-pkg/scripts.robot.py).

## ROS Integration

### Initial Setup

* Checkout crazyswarm
* Use symbolic link to include `mice-ros-pkg`: 

```
$ mkdir path/to/crazyswarm/ros_ws/src/userPackages
crazyswarm/ros_ws/src/userPackages$ ln -s /path/to/mice-ros-pkg .
```

### Run

#### Terminal 1: Select/Deselect Crazyflies to Use

* Adjust mice-ros-pkg/launch/allCrazyflies.yaml
* In Terminal 1:
```
$ source /path/to/crazyswarm/ros_ws/devel/setup.bash
$ crazyswarm/ros_ws/src/crazyswarm/scripts$ python chooser2.py --configpath ../../userPackages/mice-ros-pkg/launch/
```
* Update mice-ros-pkg/launch/run.launch if needed to select type and match CF Ids. Note that only robot.py scripts will be executed for which chooser.py enabled those robots.

#### Terminal 2: Run simulation or connect to robots

Prepare terminal:
```
$ source /path/to/crazyswarm/ros_ws/devel/setup.bash
mice-ros-pkg/scripts$ export PYTHONPATH=$PYTHONPATH:/path/to/crazyswarm/ros_ws/src/crazyswarm/scripts
```

Run simulation only (robot position will be shown in rviz):
```
mice-ros-pkg/scripts$ roslaunch mice-ros-pkg sim.launch
```

Run in lab (connection to selected crazyflies; robot position will be shown in rviz as obtained from motion capture):
```
mice-ros-pkg/scripts$ roslaunch mice-ros-pkg lab.launch
```

#### Terminal 3: Run simulation or connect to robots

Prepare terminal:
```
$ source /path/to/crazyswarm/ros_ws/devel/setup.bash
mice-ros-pkg/scripts$ export PYTHONPATH=$PYTHONPATH:/path/to/crazyswarm/ros_ws/src/crazyswarm/scripts
```

Run robot.py, one per each robot
```
mice-ros-pkg/scripts$ roslaunch mice-ros-pkg run.launch
```

## Open Issues

* Currently, there is no way to let the robots land/go home. Killing the script or pressing the red button on the joystick will let them fall down.
* How to select attacker vs. defender?
* Game Logic?
* Velocity is numerically estimated and clipped; there is no smoothing yet

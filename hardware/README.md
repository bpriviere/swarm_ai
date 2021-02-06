# Hardware Experiments for Mice

## ROS Integration

### Initial Setup

* Checkout crazyswarm
* Use symbolic link to include `mice-ros-pkg`: 

```
crazyswarm/ros_ws/src/userPackages$ ln -s /path/to/mice-ros-pkg .
```

### Run

```
crazyswarm/ros_ws/crazyswarm/scripts$ python chooser.py --basepath ../../userPackages/mice-ros-pkg/launch/
$ roslaunch mice-ros-pkg mice.launch
mice-ros-pkg/scripts$ export PYTHONPATH=$PYTHONPATH:/path/to/crazyswarm/ros_ws/src/crazyswarm/scripts
mice-ros-pkg/scripts$ examplescript.py
```

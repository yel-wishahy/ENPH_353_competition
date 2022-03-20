# ENPH_353_competition
ENPH 353 ML Project 

**Repo:** test_controller

**Description:** This repo contains a ros package that test controller for time trials that controls the robot in the enph-353 gazebo environment. 

**Author(s):** Yousif El-Wishahy


<br />

## Setting up ros package

Info for setting up and launching the competion can be found in the competion notes doc : [Competition Notes](https://docs.google.com/document/d/1FNGecS2-8v_lKdlxzUxki_ZMZeK3RGqgdYo-JMTh0OQ/edit) 

What to do when cloning this repo:
The test_controller package should be placed in the competition ros workspace within /src/ and run `catkin_make` to ensure the package is recognized.
Source setup.bash (with `source ~/<workspace name>/devel/setup.bash`) from the ros ws aften running `catkin_make`.

<br />

## Launching Controller Node

There is no ros launch file as of yet since this is a since node named `test_controller`. It subscribes to the raw camera feed at the topic: `/R1/pi_camera/image_raw`, and publishes to the topic: `/R1/cmd_vel` to move the robot.

Before launching the node:

1. Source competition setup.bash file 
    * If you follow the doc it should be `source ~/ros_ws/devel/setup.bash`)
2. Launch competition world 
    * `cd ~/ros_ws/src/2020_competition/enph353/enph353_utils/scripts`
    * `./run_sim.sh -vpg`
3. Launch score tracker (i don't really use this rn)

Launching the node takes one command if you've run `catkin_make` and sourced the correct file:

4. command:   `rosrun test_controller robot_driver.py`

<br />

## robot_driver.py
If `DEBUG=true` at the top of the file, pid error values and camera feed will be displayed.

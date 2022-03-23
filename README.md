# ENPH_353_competition
ENPH 353 ML Project 

**Repo:** test_controller_v2

**Description:** This repo contains a ros package that test controller for time trials that controls the robot in the enph-353 gazebo environment. 

**Author(s):** Yousif El-Wishahy


<br />

## Setting up repo and ros ws

**You can just clone this repo as it contains the latest competition package**

Info for setting up and launching the competion can be found in the competion notes doc : [Competition Notes](https://docs.google.com/document/d/1FNGecS2-8v_lKdlxzUxki_ZMZeK3RGqgdYo-JMTh0OQ/edit) 

What to do when cloning this repo:
1. Clone this repo
2. cd to repo
3. Run `catkin_make clean` or delete the *devel* and *build* folders
4. Run `catkin_make`


Your file tree should look like this:
![image](readme/tree.png)
<br />

## Launching Controller Node

There is no ros launch file as of yet since this is a since node named **test_controller**. It subscribes to the raw camera feed at the topic: */R1/pi_camera/image_raw*, and publishes to the topic: */R1/cmd_vel* to move the robot.

Before launching the node:

1. Source competition setup.bash file 
    * If you follow the doc it should be `source ~/ENPH_353_COMPETION/devel/setup.bash`)
2. Launch competition world 
    * `cd ~/ENPH_353_COMPETION/src/2020_competition/enph353/enph353_utils/scripts`
    * `./run_sim.sh -vpg`
3. Launch score tracker (i don't really use this rn)

Launching the node takes one command if you've run `catkin_make` and sourced the correct file:

4. command:   `rosrun test_controller robot_driver.py`

<br />

## robot_driver.py
If `DEBUG=true` at the top of the file, pid error values and camera feed will be displayed.

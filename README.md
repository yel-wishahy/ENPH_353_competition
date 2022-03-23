# ENPH_353_competition - test_controller
ENPH 353 ML Project 

**Repo:**  test_controller
**Description:** This repo contains a ros package that test controller for time trials that controls the robot in the enph-353 gazebo environment. 

**Author(s):** Yousif El-Wishahy

## Table of Contents
1. [Setting up repo and workspace](#setup)
2. [Launch Gazebo Sim](#launch)
3. [Gazebo / Robot Settings](#settings)
4. [Controller Settings](#controller_settings)


<br />

## Setting up repo and ros ws <a name="setup"></a>

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

## Launching Controller Node <a name="launch"></a>

There is no ros launch file as of yet since this is a since node named **test_controller**. It subscribes to the raw camera feed at the topic: */R1/pi_camera/image_raw*, and publishes to the topic: */R1/cmd_vel* to move the robot.

Before launching the node:

1. Source competition setup.bash file 
    * If you follow the doc it should be `source ~/ENPH_353_competition/devel/setup.bash`
2. Launch competition world 
    * `cd ~/ENPH_353_competition/src/2020_competition/enph353/enph353_utils/scripts`
    * `./run_sim.sh -vpg`
3. Launch score tracker (i don't really use this rn)

Launching the node takes one command if you've run `catkin_make` and sourced the correct file:

4. command:   `rosrun test_controller robot_driver.py`



<br />

## Gazebo Settings <a name="settings"></a>

<br />

### Changing realtime factor / simulation speed

The defualt simulation speed is ~2.5 realtime factor, to control realtime factor we change `real_time_update_rate`. The robot is much more stable at slower sim speeds!

* we can change this in the simulation menu on the fly
* or we can also change it in a settings file so that it starts up with those settings every time

In the file at `/2020_competition/enph353/enph353_gazebo/worlds/353.world` go to the bottom and find the following segment: 

```
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>10.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics> 
```

**change the number in `<real_time_update_rate>1000.0</real_time_update_rate>` from `10000.0` to `1000.0`.**

<br />

### Changing Robot Start Location and Resetting Location

For some testing cases , I found it useful to start the robot off in a specific position istead of manually having to move it every time. This is useful especially since our controller is in the early stages of development and doesn't work well in some areas of the world simulation.

**we can reset robot position to default in gazebo with:** `Ctrl+Shift+R` (this resets all models to start locations)

To change the start position of the robot, we need to modify a launch file.

Go to the file at `/2020_competition/enph353/enph353_utils/launch/robots.launch`

In this file you'll find the section:
```
	<group ns="R1">
		<param name="tf_prefix" value="R1_tf" />
		<include file="$(find adeept_awr_gazebo)/launch/adeept_awr_urdf.launch">
			<arg name="init_pose" value="-x 0.0 -y -0.85 -z 1.0 -R 0.0 -P 0.0 -Y 4.71" />
			<arg name="robot_name"  value="R1" />
		</include>
		<include file="$(find enph353_utils)/launch/desktop.launch"/>
	</group>
```

**We want to change the values in `<arg name="init_pose" value="-x 0.0 -y -0.85 -z 1.0 -R 0.0 -P 0.0 -Y 4.71" />` which are the 6 DOF of the robot.**

*Some useful start locations:*

* `"-x 0.03 -y -1.2 -z 0.1 -R 0.0 -P 0.0 -Y -0.12"` (is a stable start for the current controller because it is away from the intersections/parking blocks)



<br />

## robot_driver.py <a name="controller_settings"></a>
If `DEBUG=true` at the top of the file, pid error values and camera feed will be displayed.

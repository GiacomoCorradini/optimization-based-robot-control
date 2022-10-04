# Setup the enviroment

Open the terminal and execute the following commands:

```
sudo apt install terminator python3-numpy python3-scipy python3-matplotlib spyder3 curl

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

sudo apt-get update

sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

Configure the environment variables by adding the following lines to your file ~/.bashrc

```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
```

where <folder_containing_orc> is the folder containing the "orc" folder, which in turns contains all the python code of this class.

## Test

You can check whether the installation went fine by trying to run this python script.

```
python3 test_software.py"
```
You should see a new window appearing, displaying a robot moving somewhat randomly.

# Set the italian keyboard layout in wsl2

Configure the italian keyboard layout by adding the following lines to your file ~/.bashrc

```
setxkbmap -model pc105 -layout it -option basic
```

Check if the set went fine by typing from terminal

```
setxkbmap -query
```
 
You should see something like this:

```
rules:      evdev
model:      pc105
layout:     it
variant:    basic
```

# If you are using spyder3 (via wsl2)

The obtain the following symbol the correct combination is:

* @ = alt + altgr + ò
* \# = alt + altgr + (release alt) + à
* [] = alt + altgr + è
* {} = alt + altgr + 7
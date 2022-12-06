# Setup the enviroment

Open the terminal and execute the following commands:

```
sudo apt install terminator python3-numpy python3-scipy python3-matplotlib spyder3 curl

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

sudo apt-get update
```

If you are using Ubuntu 20.04 run:

```
sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

If you are using Ubuntu 22.04 run:

```
sudo apt install robotpkg-py310-pinocchio robotpkg-py310-example-robot-data robotpkg-urdfdom robotpkg-py310-qt5-gepetto-viewer-corba robotpkg-py310-quadprog robotpkg-py310-tsid
```

Configure the environment variables by adding the following lines to your file ~/.bashrc

To add the following command you can use the nano editor or the gedit editor:
* nano ~/.bashrc
* gedit ~/.bashrc

```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
export LOCOSIM_DIR=$HOME/orc/<folder_containing_locosim>/locosim
```

where <folder_containing_orc> is the folder containing the "orc" folder, which in turns contains all the python code of this class.

## Test

You can check whether the installation went fine by trying to run this python script.

```
python3 test_software.py
```
You should see a new window appearing, displaying a robot moving somewhat randomly.

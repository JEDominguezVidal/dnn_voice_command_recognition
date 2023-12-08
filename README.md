# dnn_voice_command_recognition

ROS package that uses Deep Learning models to detect voice commands from audio.

With this ROS package, the detected command is published in a custom message with header (including its time stamp), the command itself and its probability.

Tested on:
* OS: Ubuntu 20.04.5
* ROS: Noetic
* Nvidia Driver: 470.223 / 515.76
* CUDA: 11.0 / 11.4
* Python: 3.8.10
* Numpy: 1.23.3
* Tensorflow: 2.10.0
* Pyaudio: 0.2.13


## How to install the package

Disclaimer: This installation guide assumes ROS noetic as well as both an updated Nvidia driver and the CUDA Toolkit are already installed. To verify that this is the case you can run the following commands:
```bash
echo $ROS_DISTRO
nvidia-smi
nvcc --version
```
If any of the commands are not recognized by your machine, either ROS is not installed or the corresponding drivers are not installed or their installation path is not known.

To install this package and use it, please follow the next steps:

1. We will use a virtual environment to be able to install the corresponding dependencies without overwriting those already installed on the machine and thus avoid malfunctioning of other packages. Install the system virtualenv package: 
```bash
sudo apt-get install virtualenv
```

2. Create a folder to create your virtual environment and instantiate a new virtual environment named 'keras-voice-commands':
```bash
cd
mkdir python-virtual-environments
cd python-virtual-environments
virtualenv keras-voice-commands
```

3. Activate your new virtual environment:

```
source ~/python-virtual-environments/keras-voice-commands/bin/activate
```

4. Install the following dependencies:
```
pip install rospkg
pip install numpy==1.23.3
pip install tensorflow==2.10.0
pip install pyaudio==0.2.13
```

5. Deactivate your virtual environment:
```
deactivate
```

6. Create a new catkin workspace (or jump to step next step in case you already have your workspace created):
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
source devel/setup.bash
```

7. Inside your catkin workspace, copy this repository and compile its custom messages:
```
cd ~/catkin_ws/src
git clone https://github.com/JEDominguezVidal/dnn_voice_command_recognition
cd ..
catkin_make
```


### Potential Installation Issues
Two known bugs have been detected when installing the Pyaudio library:
1. The error "ERROR: Could not build wheels for pyaudio which use PEP 517 and cannot be installed directly" has been resolved by updating pip to the latest version (23.3.1 at the time of writing this document):
```
pip3 install --upgrade pip
```

2. The error "ERROR: Could not build wheels for pyaudio, which is required to install pyproject.toml-based projects" has been resolved by installing the PortAudio library at system level:
```
sudo apt-get install portaudio19-dev
pip install pyaudio==0.2.13
```



## How to use the package

Follow this steps to run this ROS package:

1. Activate a virtual environment with the necessary dependancies installed. If you followd the installation guide, you can use the following command:
```
source ~/python-virtual-environments/keras-voice-commands/bin/activate
```

2. Move to the active catkin workspace. If you follow the installation guide, it should be ~/catkin_ws/:
```bash
roscd && cd ../src
```

3. Run the package executing the following launch file:
```
roslaunch dnn_voice_command_recognition voice_recognition.launch
```

4. Open a rqt_reconfigure window in a new terminal to tune the available params:
```
rosrun rqt_reconfigure rqt_reconfigure
```


# dnn_voice_command_recognition

ROS package that uses OpenAI's Whisper to detect voice commands from audio.

With this ROS package, the detected command is published in a custom message with header (including its time stamp), the command itself, and its probability.

Tested on:
* OS: Ubuntu 20.04.5
* ROS: Noetic
* Nvidia Driver: 470.223 / 515.76
* CUDA: 11.0 / 11.4
* Python: 3.8.10
* Numpy: 1.24.4
* Pytorch: 2.1.0
* Pyaudio: 0.2.14


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
virtualenv voice-commands
```

3. Activate your new virtual environment:

```bash
source ~/python-virtual-environments/voice-commands/bin/activate
```

4. Create a new catkin workspace (or jump to step next step in case you already have your workspace created):
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
source devel/setup.bash
```

5. Inside your catkin workspace, copy this repository and compile its custom messages:
```bash
cd ~/catkin_ws/src
git clone https://github.com/JEDominguezVidal/dnn_voice_command_recognition
cd ..
catkin_make
```

6. Install the dependencies:
```bash
pip install -r requirements.txt
```



### Potential Installation Issues
Two known bugs have been detected when installing the Pyaudio library:
1. The error "ERROR: Could not build wheels for pyaudio which use PEP 517 and cannot be installed directly" has been resolved by updating pip to the latest version (23.3.1 at the time of writing this document):
```bash
pip3 install --upgrade pip
```

2. The error "ERROR: Could not build wheels for pyaudio, which is required to install pyproject.toml-based projects" has been resolved by installing the PortAudio library at system level:
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio==0.2.14
```


## How to use the package

Follow this steps to run this ROS package:

1. Activate a virtual environment with the necessary dependancies installed. If you followd the installation guide, you can use the following command:
```bash
source ~/python-virtual-environments/voice-commands/bin/activate
```

2. Move to the active catkin workspace. If you follow the installation guide, it should be ~/catkin_ws/:
```bash
roscd && cd ../src
```

3. Run the package executing the following launch file:
```bash
roslaunch dnn_voice_command_recognition voice_recognition.launch
```

4. Open a rqt_reconfigure window in a new terminal to tune the available params:
```bash
rosrun rqt_reconfigure rqt_reconfigure
```

### Audio Processing Parameters

These parameters control the overlapping window processing for voice command detection:

- **buffer_seconds** (default: 2.0):  
  Total duration of the audio buffer in seconds. This determines how much audio history is maintained.

- **window_seconds** (default: 1.0):  
  Duration of the processing window in seconds. This is the audio segment sent to Whisper for transcription.

- **step_seconds** (default: 0.5):  
  Step size between processing windows in seconds. This creates 50% overlap by default to ensure no commands are missed at boundaries.

> **Example**: With buffer_seconds=2.0, window_seconds=1.0, and step_seconds=0.5, the system processes overlapping 1-second windows every 0.5 seconds, ensuring every audio sample is analyzed twice.

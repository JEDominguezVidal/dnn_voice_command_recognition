#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified for Whisper integration
"""

import sys
import os
from threading import Thread
import time
import numpy as np

from dnn_voice_command_recognition.cfg import NodeConfig as Config
import rospy
from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from std_msgs.msg import String, Header
from dnn_voice_command_recognition.msg import dnn_voice_command
from recording_helper import AudioRecorder  # Updated for overlapping processing
from whisper_interface import WhisperInterface, map_to_command


def main_thread(arg):
    """
    Main processing thread for voice command recognition with overlapping windows.
    
    Continuously records audio in steps, updates circular buffer,
    processes overlapping windows with Whisper, and publishes commands.
    
    Args:
        arg: DNN_Voice_Command_Recognition_Node instance
    """
    while not rospy.is_shutdown():
        try:
            # Read new audio samples for this step
            new_samples = arg.recorder.read_samples(arg.step_samples)
            
            # Update circular buffer
            end_index = arg.write_index + arg.step_samples
            if end_index <= arg.buffer_samples:
                arg.audio_buffer[arg.write_index:end_index] = new_samples
            else:
                # Handle wrap-around
                first_part = arg.buffer_samples - arg.write_index
                arg.audio_buffer[arg.write_index:] = new_samples[:first_part]
                arg.audio_buffer[:end_index - arg.buffer_samples] = new_samples[first_part:]
            
            # Update write index with wrap-around
            arg.write_index = (arg.write_index + arg.step_samples) % arg.buffer_samples
            
            # Extract processing window (most recent window_samples)
            start_index = arg.write_index - arg.window_samples
            if start_index < 0:
                # Wrap around case
                window = np.concatenate((
                    arg.audio_buffer[start_index:],
                    arg.audio_buffer[:arg.write_index]
                ))
            else:
                # Simple contiguous case
                window = arg.audio_buffer[start_index:arg.write_index]
            
            # Process window with Whisper
            start_time = time.time()
            transcript = arg.whisper.transcribe(window)
            latency = time.time() - start_time
            
            # Log latency and check for fallback
            if latency > 0.5 and arg.whisper.model_size != "tiny":
                rospy.logwarn(f"High latency ({latency:.2f}s), falling back to tiny model")
                arg.whisper = WhisperInterface(model_size="tiny")
            
            command = map_to_command(transcript, arg.command_list)
            
            # Publish detected command
            arg.dnn_voice_command.header = Header(stamp=rospy.Time.now())
            arg.dnn_voice_command.command = command
            arg.dnn_voice_command.probability = 1.0  # Fixed value for now
            arg.publisher_voice_command.publish(arg.dnn_voice_command)
            
            rospy.loginfo(f"Detected: {command} (transcript: '{transcript}')")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                rospy.logerr("GPU memory overflow, reducing model size")
                arg.whisper = WhisperInterface(model_size="tiny")
            else:
                rospy.logerr(f"Error in processing: {e}")
        
        arg.rate.sleep()

class DNN_Voice_Command_Recognition_Node:
    """
    ROS node for real-time voice command recognition using Whisper.
    
    Handles audio input, Whisper transcription, command mapping, and ROS communication.
    """
    
    def __init__(self):
        """
        Initialize the voice command recognition node with overlapping audio processing.
        
        Sets up:
        - ROS parameters and configuration
        - Audio processing with circular buffer
        - Whisper speech recognition model
        - ROS publishers and services
        - Dynamic reconfigure server
        """
        # get the main thread desired rate of the node
        self.rate_value = rospy.get_param('~rate', 10)
        self.rate = rospy.Rate(self.rate_value)

        rospy.loginfo(rospy.get_caller_id() + ": Starting Whisper Voice Command Recognition Node")

        # Audio configuration
        self.RATE = 16000  # Fixed sample rate for Whisper
        
        # New parameters for overlapping windows
        self.buffer_seconds = 2.0    # Total buffer duration
        self.window_seconds = 1.0    # Processing window duration
        self.step_seconds = 0.5      # Step size (50% overlap)
        
        # Calculate sample sizes
        self.buffer_samples = int(self.buffer_seconds * self.RATE)
        self.window_samples = int(self.window_seconds * self.RATE)
        self.step_samples = int(self.step_seconds * self.RATE)
        
        # Initialize audio buffer
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.int16)
        self.write_index = 0
        
        # Create audio recorder
        self.recorder = AudioRecorder(rate=self.RATE, frames_per_buffer=800)
        self.recorder.start()
        
        # Message for publishing
        self.dnn_voice_command = dnn_voice_command()
        
        # Default command list
        self.command_list = rospy.get_param('~commands', 
            ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'])
        
        # Initialize Whisper
        self.whisper = WhisperInterface(model_size="small")
        
        # Create topic publisher
        self.publisher_voice_command = rospy.Publisher("~publisher_voice_command", dnn_voice_command, queue_size=1)
        
        rospy.loginfo(rospy.get_caller_id() + ": Node initialization complete")
        
        # Create dynamic_reconfigure server AFTER all attributes are initialized
        self.dyn_reconf_server = DynamicReconfigureServer(Config, self.dyn_reconf_callback)
        rospy.loginfo(rospy.get_caller_id() + ": Dynamic reconfigure server started")

        rospy.loginfo(rospy.get_caller_id() + ": Init DNN_Voice_Command_Recognition_Node Done")

        # Start main processing thread
        self.thread = Thread(target=main_thread, args=(self,))
        self.thread.start()

    def dyn_reconf_callback(self, config, level):
        """
        Dynamic reconfigure callback to update node parameters.
        
        Args:
            config: New configuration
            level: Bitmask describing changed parameters
            
        Returns:
            Config: Updated configuration
        """
        # Safe parameter access with fallbacks
        rate = config.get('rate', 10)
        frames_per_buffer = config.get('FRAMES_PER_BUFFER', 4000)
        command_list_str = config.get('command_list', "background_noise,down,go,left,no,off,on,right,stop,unknown,up,yes")
        buffer_seconds = config.get('buffer_seconds', 2.0)
        window_seconds = config.get('window_seconds', 1.0)
        step_seconds = config.get('step_seconds', 0.5)
        
        self.rate = rospy.Rate(rate)
        rospy.loginfo(f"{rospy.get_caller_id()}: Reconfigure Request: rate={rate}, FRAMES_PER_BUFFER={frames_per_buffer}, command_list={command_list_str}, buffer_seconds={buffer_seconds}, window_seconds={window_seconds}, step_seconds={step_seconds}")
        
        # Update parameters
        self.FRAMES_PER_BUFFER = frames_per_buffer
        
        # Update overlapping window parameters if changed
        if (buffer_seconds != self.buffer_seconds or 
            window_seconds != self.window_seconds or 
            step_seconds != self.step_seconds):
            
            self.buffer_seconds = buffer_seconds
            self.window_seconds = window_seconds
            self.step_seconds = step_seconds
            
            # Recalculate sample sizes
            self.buffer_samples = int(self.buffer_seconds * self.RATE)
            self.window_samples = int(self.window_seconds * self.RATE)
            self.step_samples = int(self.step_seconds * self.RATE)
            
            # Reinitialize audio buffer
            self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.int16)
            self.write_index = 0
            rospy.loginfo(f"Updated window parameters: buffer={self.buffer_samples} samples, window={self.window_samples} samples, step={self.step_samples} samples")
        
        # Update command list
        if command_list_str != ",".join(self.command_list):
            self.command_list = [cmd.strip() for cmd in command_list_str.split(",")]
            rospy.loginfo(f"Updated command list: {self.command_list}")
        
        return config

def main(args):
    """
    Main entry point for the ROS node.
    
    Initializes ROS and starts the voice command recognition node.
    
    Args:
        args: Command line arguments
    """
    rospy.init_node('dnn_voice_command_recognition', anonymous=False)
    tn = DNN_Voice_Command_Recognition_Node()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)

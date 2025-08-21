#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified for Whisper integration
"""

import sys
import os
from threading import Thread
import time

from dnn_voice_command_recognition.cfg import NodeConfig as Config
import rospy
from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from std_msgs.msg import String, Header
from dnn_voice_command_recognition.msg import dnn_voice_command
from recording_helper import record_chunk_audio, convert_frames_to_audio
from whisper_interface import WhisperInterface, map_to_command

import numpy as np


def main_thread(arg):
    """
    Main processing thread for voice command recognition.
    
    Continuously records audio chunks, processes them with Whisper,
    and publishes detected commands.
    
    Args:
        arg: DNN_Voice_Command_Recognition_Node instance
    """
    while not rospy.is_shutdown():
        # Async audio recording - capture next chunk while processing current
        if not hasattr(arg, 'next_frames'):
            arg.next_frames = record_chunk_audio(arg.frames, arg.seconds, arg.FRAMES_PER_BUFFER, arg.RATE)
        else:
            arg.next_frames = record_chunk_audio(arg.next_frames, arg.seconds, arg.FRAMES_PER_BUFFER, arg.RATE)
        
        # Process current frames if buffer is full
        if len(arg.frames) >= (int(arg.RATE / arg.FRAMES_PER_BUFFER * arg.seconds) - 1):
            # Process current chunk
            audio = convert_frames_to_audio(arg.frames)
        
            try:
                start_time = time.time()
                transcript = arg.whisper.transcribe(audio)
                latency = time.time() - start_time
                
                # Log latency and check for fallback
                if latency > 0.5 and arg.whisper.model_size != "tiny":
                    rospy.logwarn(f"High latency ({latency:.2f}s), falling back to tiny model")
                    arg.whisper = WhisperInterface(model_size="tiny")
                
                command = map_to_command(transcript, arg.command_list)
                command_prob = 1.0  # Whisper doesn't provide per-command probability
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    rospy.logerr("GPU memory overflow, reducing model size")
                    arg.whisper = WhisperInterface(model_size="tiny")
                    return
                raise

            # Publish detected command
            arg.dnn_voice_command.header = Header(stamp=rospy.Time.now())
            arg.dnn_voice_command.command = command
            arg.dnn_voice_command.probability = 1.0  # Fixed value for now
            arg.publisher_voice_command.publish(arg.dnn_voice_command)
                
            rospy.loginfo(f"Detected: {command} (transcript: '{transcript}')")
                
            # Swap buffers for next iteration
            arg.frames = arg.next_frames

    arg.rate.sleep()

class DNN_Voice_Command_Recognition_Node:
    """
    ROS node for real-time voice command recognition using Whisper.
    
    Handles audio input, Whisper transcription, command mapping, and ROS communication.
    """
    
    def __init__(self):
        """
        Initialize the voice command recognition node.
        
        Sets up:
        - ROS parameters and configuration
        - Audio processing buffers
        - Whisper speech recognition model
        - ROS publishers and services
        - Dynamic reconfigure server
        """
        # get the main thread desired rate of the node
        self.rate_value = rospy.get_param('~rate', 10)
        self.rate=rospy.Rate(self.rate_value)

        rospy.loginfo(rospy.get_caller_id() + ": Starting Whisper Voice Command Recognition Node")

        # Audio configuration
        self.frames = []
        self.seconds = 0.5  # Audio chunk duration (seconds)
        self.FRAMES_PER_BUFFER = 4000
        self.RATE = 16000  # Fixed sample rate for Whisper
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
        
        self.rate = rospy.Rate(rate)
        rospy.loginfo(f"{rospy.get_caller_id()}: Reconfigure Request: rate={rate}, FRAMES_PER_BUFFER={frames_per_buffer}, command_list={command_list_str}")
        
        # Update parameters
        self.FRAMES_PER_BUFFER = frames_per_buffer
        
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

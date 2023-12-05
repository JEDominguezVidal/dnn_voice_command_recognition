#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 19 2023

@author: jdominguez@iri.upc.edu
"""

#from venv_utils import activate_virtual_env
#venv_status = activate_virtual_env()

import sys
import os
from threading import Thread

from dnn_voice_command_recognition.cfg import NodeConfig as Config

#########################################
# Add any ROS dependency
#########################################
import rospy
from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from std_msgs.msg import String, Header, Bool, Float64
from std_srvs.srv import Empty,EmptyResponse
from dnn_voice_command_recognition.msg import dnn_voice_command

#########################################
# Add any non ROS dependency
#########################################
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from recording_helper import record_audio, record_chunk_audio, convert_frames_to_audio, terminate
from tf_helper import preprocess_audiobuffer
from utils.models import ResnetBlock, ResNet18

import time


def main_thread(arg):
  while not rospy.is_shutdown():
    #########################################
    # DNN feed forward
    #########################################

    if (arg.previous_selected_model != arg.selected_model):
        arg.model_change_requested = True

    if (arg.model_change_requested):
        arg.model_change_requested = False
        # Load new selected model
        if (arg.selected_model == 0): # 8 labels (small) model selected
            arg.commands = arg.commands_8_labels
            arg.loaded_model = tf.keras.models.load_model(arg.model_path_0, compile=False)
            arg.loaded_model.summary()
            arg.previous_selected_model = arg.selected_model
        elif (arg.selected_model == 1): # 8 labels (medium) model selected
            arg.commands = arg.commands_8_labels
            arg.loaded_model = tf.keras.models.load_model(arg.model_path_1, compile=False)
            arg.loaded_model.summary()
            arg.previous_selected_model = arg.selected_model
        elif (arg.selected_model == 2): # 12 labels (small) model selected
            arg.commands = arg.commands_12_labels
            arg.loaded_model = tf.keras.models.load_model(arg.model_path_2, compile=False)
            arg.loaded_model.summary()
            arg.previous_selected_model = arg.selected_model
        elif (arg.selected_model == 3): # 12 labels (medium) model selected
            arg.commands = arg.commands_12_labels
            arg.loaded_model = tf.keras.models.load_model(arg.model_path_3, compile=False)
            arg.loaded_model.summary()
            arg.previous_selected_model = arg.selected_model
        elif (arg.selected_model == 4): # 35 labels (small) model selected
            arg.commands = arg.commands_35_labels
            arg.loaded_model = tf.keras.models.load_model(arg.model_path_4, compile=False)
            arg.loaded_model.summary()
            arg.previous_selected_model = arg.selected_model
        elif (arg.selected_model == 5): # 35 labels (medium) model selected
            arg.commands = arg.commands_35_labels
            arg.loaded_model = tf.keras.models.load_model(arg.model_path_5, compile=False)
            arg.loaded_model.summary()
            arg.previous_selected_model = arg.selected_model
        elif (arg.selected_model == 6): # 35 labels (large) model selected
            arg.commands = arg.commands_35_labels
            arg.loaded_model = ResNet18(36)
            arg.loaded_model.build(input_shape = (None,129,124,1))
            arg.loaded_model.load_weights(arg.model_path_6)
            arg.loaded_model.summary()
            arg.previous_selected_model = arg.selected_model
        else:
            rospy.logwarn("WARNING: Selected model not implemented")

    else:
        arg.frames = record_chunk_audio(arg.frames, arg.seconds, arg.FRAMES_PER_BUFFER, arg.RATE)
        if (len(arg.frames) > (int(arg.RATE / arg.FRAMES_PER_BUFFER * arg.seconds) - 1)):
            command, command_prob = predict_command(arg.loaded_model, arg.commands, arg.frames, arg.debug)

            arg.dnn_voice_command.header = Header(stamp=rospy.Time.now())
            arg.dnn_voice_command.command = String()
            arg.dnn_voice_command.probability = Float64()

            if (arg.debug):
                if (command_prob > arg.prob_threshold):
                    if command == "down":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                    elif command == "go":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                    elif command == "left":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                    elif command == "no":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                    elif command == "right":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                    elif command == "stop":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                    elif command == "up":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                    elif command == "yes":
                        print("-------------")
                        print("Predicted label:", command)
                        print("Predicted label prob.:", command_prob)
                else:
                    print("WARNING: Low probability or unknown command")
                    print("Predicted label:", command)
                    print("Predicted label prob.:", command_prob)

            arg.dnn_voice_command.command = command
            arg.dnn_voice_command.probability = command_prob
            arg.publisher_voice_command.publish(arg.dnn_voice_command)

    arg.rate.sleep()

class DNN_Voice_Command_Recognition_Node:

  def __init__(self):
    # get the main thread desired rate of the node
    self.rate_value = rospy.get_param('~rate', 10)
    self.rate=rospy.Rate(self.rate_value)

    #Create dynamic_reconfigure server
    self.dyn_reconf_server = DynamicReconfigureServer(Config,self.dyn_reconf_callback)

    rospy.loginfo(rospy.get_caller_id() + ": Init DNN_Voice_Command_Recognition_Node")

    self.debug = False
    self.frames = []
    self.seconds = 1
    self.FRAMES_PER_BUFFER = 4000
    self.RATE = 16000
    self.dnn_voice_command = dnn_voice_command()
    absolute_path = os.path.dirname(__file__)
    self.model_path_0 = os.path.join(absolute_path, "models/audio_commands_recognition_8_lables_accuracy_0.9775_small.h5")
    self.model_path_1 = os.path.join(absolute_path, "models/audio_commands_recognition_8_lables_accuracy_0.9814_medium.h5")
    self.model_path_2 = os.path.join(absolute_path, "models/audio_commands_recognition_12_lables_WITH_BG_accuracy_0.9612_small.h5")
    self.model_path_3 = os.path.join(absolute_path, "models/audio_commands_recognition_12_lables_WITH_BG_accuracy_0.9724_medium.h5")
    self.model_path_4 = os.path.join(absolute_path, "models/audio_commands_recognition_35_lables_WITH_BG_accuracy_0.9519_small.h5")
    self.model_path_5 = os.path.join(absolute_path, "models/audio_commands_recognition_35_lables_WITH_BG_accuracy_0.9563_medium.h5")
    self.model_path_6 = os.path.join(absolute_path, "models/audio_commands_recognition_35_lables_WITH_BG_accuracy_0.9713_large_resnet.h5")
    self.commands_8_labels = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
    self.commands_12_labels = ['background_noise', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'unknown', 'up', 'yes']
    self.commands_35_labels = ['background_noise', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    self.selected_model = 0
    self.previous_selected_model = 0
    self.model_change_requested = False

    # Load DNN model
    self.prob_threshold = 0.9
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # default model: 8 labels (small)
    self.commands = self.commands_8_labels
    self.loaded_model = tf.keras.models.load_model(self.model_path_0, compile=False)
    self.loaded_model.summary()

    # Create topic publisher (dnn_voice_command.msg)
    self.publisher_voice_command = rospy.Publisher("~publisher_voice_command", dnn_voice_command, queue_size=1)

    rospy.loginfo(rospy.get_caller_id() + ": Init DNN_Voice_Command_Recognition_Node Done")

    # start main thread
    self.thread = Thread(target = main_thread, args = (self, ))
    self.thread.start()

  def dyn_reconf_callback(self, config, level):
    self.rate=rospy.Rate(config.rate)

    rospy.loginfo(rospy.get_caller_id() +": "+ """Reconfigure Request: {rate}, {debug}, {prob_threshold}, {SAMPLING_RATE}, {FRAMES_PER_BUFFER}, {model}""".format(**config))
    self.debug = config.debug
    self.prob_threshold = config.prob_threshold
    self.RATE = config.SAMPLING_RATE
    self.FRAMES_PER_BUFFER = config.FRAMES_PER_BUFFER
    self.selected_model = config.model

    return config



def predict_command(loaded_model, commands, frames, debug):
    audio = convert_frames_to_audio(frames)
    spec = preprocess_audiobuffer(audio)
    start = time.time()
    prediction = loaded_model(spec)
    if (debug):
        print(f'Inference time [s]: {time.time() - start}')
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    prediction_numpy = (tf.nn.softmax(prediction[0])).numpy()
    command_prob = prediction_numpy[label_pred[0]]

    return command, command_prob



def main(args):

  rospy.init_node('dnn_voice_command_recognition', anonymous=False)
  tn = DNN_Voice_Command_Recognition_Node()
  rospy.spin()

if __name__ == '__main__':
  main(sys.argv)

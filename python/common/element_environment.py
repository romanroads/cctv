# Copyright 2019 - 2024 The ROMAN ROADS Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import gym
from gym import spaces
import numpy as np
import random

try:
    from pynng import Sub0, Pair0
except ModuleNotFoundError:
    pass


class ElementEnvironment(gym.Env):
    def __init__(self, e_id, log_ip_address='127.0.0.1', log_port=5555, control_port=5556):
        super(ElementEnvironment, self).__init__()

        # Note: we support several behaviors for your to control the ego car including lane keep, turns, u-turn...
        self.num_behaviors = 2
        self.action_space = spaces.MultiDiscrete([self.num_behaviors])

        # Note: you can determine how many features are needed as input observation
        self.num_input_feature = 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.num_input_feature,), dtype=np.uint8)

        self.env_id = e_id

        self.observation = np.zeros([self.num_input_feature]).astype(np.float32)
        self.reward = 0.
        self.done = False
        self.info = {}

        self.topic = "my_ego_vehicle"
        self.header_bytes = str.encode(self.topic)
        self.header_length = len(self.header_bytes)

        log_ip_port = "tcp://%s:%s" % (log_ip_address, log_port)
        control_ip_port = "tcp://%s:%s" % (log_ip_address, control_port)
        self.socket_nanomsg = Sub0(dial=log_ip_port, recv_timeout=10000, send_timeout=10000)
        self.socket_nanomsg.subscribe(self.topic)
        self.socket_control = Pair0(dial=control_ip_port, recv_timeout=100, send_timeout=100)

        self.initialize_element()

        self.time_to_restart = 5
        self.initial_time = None

    def initialize_element(self):
        """
        Note this initialization command will start spawning traffic and select the specified human demonstrators
        for imitation learning
        :return:
        """
        init_command = {
            "StartLearning": True,
            "AgentID": 1854
        }
        msg = json.dumps(init_command).encode('unicode_escape')
        self.socket_control.send(msg)

    def step(self, action):
        packet = self.socket_nanomsg.recv()
        payload = packet[self.header_length:]
        payload = payload.decode('unicode_escape')
        dictionary = json.loads(payload)
        timestamp = int(dictionary["Timestamp"])

        if self.initial_time is None:
            self.initial_time = timestamp

        self.done = dictionary["Restart"] == "True"
        if self.done:
            return self.observation, self.reward, self.done, self.info

        t = (timestamp - self.initial_time) / 1000.

        print("[INFO] timestamp %s" % timestamp)

        # Note: craft your own reward function, observations here using data read from Element
        # ......
        self.reward = random.uniform(-1, 1)

        # Note: generate your actions, decisions here based upon your policy
        # ......
        decision_command = {
            "Timestamp": timestamp,
            "Behavior": "KeepLane",
            "Duration": 1.0,
            "Acceleration": 1.2
        }

        # Note: we send out your decision sequence to Element environment to control the ego car
        msg = json.dumps(decision_command).encode('unicode_escape')
        self.socket_control.send(msg)

        return self.observation, self.reward, self.done, self.info

    def reset(self):
        """
        this reset should block until env is ready again after reset
        :return:
        """
        while True:
            packet = self.socket_nanomsg.recv()
            payload = packet[self.header_length:]
            payload = payload.decode('unicode_escape')
            dictionary = json.loads(payload)
            restart = dictionary["Restart"] == "True"
            if not restart:
                break

        return self.observation

    def render(self):
        pass

    def close(self):
        decision_command = {
            "Restart": True
        }
        msg = json.dumps(decision_command).encode('unicode_escape')
        self.socket_control.send(msg)
        self.socket_nanomsg.close()
        self.socket_control.close()

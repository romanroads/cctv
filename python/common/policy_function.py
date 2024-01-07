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
import gym

try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


class MyPolicy(object):
    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers):
        assert isinstance(ob_space, gym.spaces.Box)
        # Note: build your own NN based non-linear policy function here, preferably using tensorflow
        # ......

    def act(self, stochastic, ob):
        # Note: invoke tensorflow session run to evaluate your action network below
        action, value = 0, 0
        return action, value

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


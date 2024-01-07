# Copyright 2019 - 2021 The ROMAN ROADS Developers. All Rights Reserved.
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
r"""Sample code to setup RL, IL training environment using Element platform
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from policy_function import MyPolicy
from element_environment import ElementEnvironment
from episode_segment_generator import segment_generator


def train():
    env_id = 'Element-ram-v0'
    env = ElementEnvironment(env_id)

    iters_so_far = 0
    max_iters = 2
    timesteps_per_batch = 80

    pi = MyPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space, hid_size=2, num_hid_layers=2)
    seg_gen = segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    while True:
        if max_iters and iters_so_far >= max_iters:
            break

        seg = seg_gen.__next__()
        observation, action, value_function = seg["ob"], seg["ac"], seg["vpred"]

        # Note: updating your policy below
        # .....

        iters_so_far += 1

    env.close()


def main():
    try:
        train()
    except:
        pass


if __name__ == "__main__":
    main()

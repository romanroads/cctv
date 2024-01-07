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
r"""Sample code to test reading data and writing commands using Element platform
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

try:
    from pynng import Sub0, Pair0
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass

LOG_IP = '127.0.0.1'
LOG_PORT = 5555
CONTROL_PORT = 5556
LOG_IP_PORT = "tcp://%s:%s" % (LOG_IP, LOG_PORT)
CONTROL_IP_PORT = "tcp://%s:%s" % (LOG_IP, CONTROL_PORT)
TIME_TO_CHANGE_LANE = 5
TIME_TO_RESTART = 15


def test_using_nanomsg_socket():
    topic = "my_ego_vehicle"
    header_bytes = str.encode(topic)
    header_length = len(header_bytes)
    duration_change_lane = 1.5
    decision_sequence = []

    try:
        socket_nanomsg = Sub0(dial=LOG_IP_PORT, recv_timeout=10000, send_timeout=10000)
        socket_nanomsg.subscribe(topic)

        socket_control = Pair0(dial=CONTROL_IP_PORT, recv_timeout=100, send_timeout=100)

        while True:
            packet = socket_nanomsg.recv()
            payload = packet[header_length:]
            payload = payload.decode('unicode_escape')
            dictionary = json.loads(payload)
            timestamp = int(dictionary["Timestamp"])

            if "initial_time" not in locals():
                initial_time = timestamp

            print("[INFO] timestamp %s: %s" % (timestamp, dictionary))
            decision_command = {
                "Timestamp": timestamp,
                "Behavior": "KeepLane",
                "Duration": 1.0,
                "Acceleration": 1.2
            }

            t = (timestamp - initial_time) / 1000.
            if TIME_TO_CHANGE_LANE < t < TIME_TO_CHANGE_LANE + duration_change_lane:
                decision_command["Behavior"] = "LeftLaneChange"
                decision_command["Duration"] = duration_change_lane

            if t > TIME_TO_RESTART:
                decision_command["Restart"] = True
                initial_time = timestamp
                msg = json.dumps(decision_command).encode('unicode_escape')
                socket_control.send(msg)
                break

            decision_sequence.append((t, decision_command["Behavior"]))

            msg = json.dumps(decision_command).encode('unicode_escape')
            socket_control.send(msg)

        socket_nanomsg.close()
        socket_control.close()
    except:
        print("[ERROR] caught exception: ", sys.exc_info()[0])
        pass

    plot_decision_sequence(decision_sequence)


def plot_decision_sequence(decision_sequences):
    plt.rcParams["font.family"] = "Times New Roman"
    dictionary_decision_names = {"KeepLane": 0, "LeftLaneChange": 1, "RightLaneChange": 2}
    decision_sequence = [dictionary_decision_names[d[1]] for d in decision_sequences]
    time = [d[0] for d in decision_sequences]
    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(111)

    ax1.set_xlabel("Time [s]", fontsize=30)
    ax1.set_ylabel("Decision Sequence", fontsize=30)
    ax1.plot(time, decision_sequence, 'b-o', linewidth=5)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["KeepLane", "LeftLaneChange", "RightLaneChange"])
    plt.tight_layout()
    plt.show()


def main():
    test_using_nanomsg_socket()


if __name__ == "__main__":
    main()

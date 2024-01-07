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
r"""Sample code to run training using Element platform
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket

LOG_IP = '127.0.0.1'
LOG_PORT = 5555
CONTROL_PORT = 5556
BUFFER_SIZE = 1024


def test_using_simple_socket():
    socket_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_log.connect((LOG_IP, LOG_PORT))
        socket_control.connect((LOG_IP, CONTROL_PORT))

        while True:
            packet = socket_log.recv(BUFFER_SIZE)
            if not packet:
                break
            payload = packet[:]
            print(payload)

            control_signal = "0:1:0:0:0:0:0:0:0:0:0"
            socket_control.send(control_signal.encode())

    except ConnectionRefusedError:
        print("Connection refused....")


def main():
    test_using_simple_socket()


if __name__ == "__main__":
    main()

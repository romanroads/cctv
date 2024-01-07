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
import os
import sys
import platform
import logging

from constants import *


def load_decryption_key():
    os_name = platform.system()
    if "Windows" in os_name:
        repo_path = REPO_PATH_WINDOWS
    else:
        repo_path = REPO_PATH

    user_sample_folder = os.path.join(repo_path, USER_SAMPLE_FOLDER)
    python_folder = os.path.join(user_sample_folder, PYTHON_FOLDER)
    cred_folder = os.path.join(python_folder, CREDENTIAL_FOLDER)
    key_file = os.path.join(cred_folder, DECRYPTION_KEY_FILE)

    if not os.path.exists(key_file):
        logging.error("load_decryption_key: key_file does not exist")
        sys.exit(0)

    with open(key_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1, "load_decryption_key: key file content corrupted"
        line = lines[0]
        return line.strip()

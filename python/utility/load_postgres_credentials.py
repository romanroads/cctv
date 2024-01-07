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
import logging


def load_postgres_credentials(credential_file):
    if not os.path.exists(credential_file):
        logging.error("load_postgres_credentials: postgres credential file does not exist")
        return

    with open(credential_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2, "load_postgres_credentials: postgres credential file content corrupted"
        line = lines[1]
        tokens = line.strip().split(",")
        assert len(tokens) == 5, "load_postgres_credentials: postgres credential file content corrupted"
        user_name, password, endpoint, port, db_name = tokens
        return user_name, password, endpoint, port, db_name


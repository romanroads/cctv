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
import tensorflow.compat.v1 as tfc

from common.aes_cipher import AESCipher
from constants import *


def load_graph(frozen_graph_filename, is_decryption_needed=False, decryption_key=None):
    with tfc.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tfc.GraphDef()
        file_byte_data = f.read()

        if is_decryption_needed:
            cipher = AESCipher(decryption_key)
            file_byte_data = cipher.decrypt(file_byte_data)

        graph_def.ParseFromString(file_byte_data)

    with tfc.Graph().as_default() as graph:
        # Note: The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tfc.import_graph_def(graph_def, name="")

    return graph

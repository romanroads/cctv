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
import hashlib

try:
    from Crypto import Random
    from Crypto.Cipher import AES
except:
    raise Exception('Install Crypto! \n pip install pycrypto')

from constants import *


class AESCipher(object):
    def __init__(self, _key):
        self.bs = 16  # bytes, 128 bits
        self.key = hashlib.sha256(_key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(raw)

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))  # .decode('utf-8')

    def _pad(self, s):
        return s + str.encode((self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs))\

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]

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

AWS_REGION = "us-west-2"
AWS_REGION_CN = "cn-north-1"

AWS_S3_VIDEO_BUCKET = "user-upload-data"

# Note: /home/element_cctv_symbolic_link is a symbolic link that points to repo folder
REPO_PATH = "/home/element_cctv_symbolic_link"

# Note: C:\Users\element_cctv_symbolic_link is a symbolic link that needs to be created on your machine
REPO_PATH_WINDOWS = r"C:\Users\element_cctv_symbolic_link"

USER_SAMPLE_FOLDER = "user_sample"
PYTHON_FOLDER = "python"

CREDENTIAL_FOLDER = "credentials"

AWS_CREDENTIAL_FILE = "aws_accessKeys.csv"
SQL_CREDENTIAL_FILE = "behavioral_database_sql.csv"
SQL_CREDENTIAL_FILE_CN = "behavioral_database_sql_cn.csv"
DECRYPTION_KEY_FILE = "decryption_key.csv"

CACHE_DATA_PATH = "data"

NUM_FEATURE_POINTS = 0

WINDOW_NAME = "ROMAN ROADS RCIU"
WIDTH_WINDOW = 800
HEIGHT_WINDOW = 600
IMAGE_WIDTH_ENDTER_DNN = 800
IMAGE_HEIGHT_ENDTER_DNN = 600
WINDOW_START_WIDTH = 100
WINDOW_START_HEIGHT = 100

OUTPUT_TENSOR_NAME = "hybrid_boxes_fixed_len:0"

NUM_FIXED_AGENTS = 100
NUM_FIXED_FEATURES = 790

WHITE = (255, 255, 255)
BLUE_BRG = (255, 0, 0)

NUM_MASK_POINTS = 28
THRESHOLD_CONFIDENCE_SCORE = 0.9

ESCAPE_KEY = 27


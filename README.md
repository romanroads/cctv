# ELEMENT - CCTV Project

Sample code for users

## Setup

### OS Ubuntu
#### Create symbolic link to your repo folder
    - sudo ln -s /home/your_repo_folder /home/element_cctv_symbolic_link

#### Create virtual environment
    - conda --version
        conda 4.10.3
    - cd python
    - conda env create --file environment_ubuntu.yml

#### Misc

#### FAQ
    - pip install slow in China, use -i to select index source in China
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple your_package
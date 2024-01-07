# ELEMENT

Sample code for users

## Setup

### Windows 10
#### Create symbolic link to your repo folder
    - MKLINK /J "C:\Users\element_symbolic_link" "C:\Users\your_repo_folder"
    - conda env create --file environment_windows.yml
    
### Ubuntu
#### Create symbolic link to your repo folder
    - sudo ln -s /home/your_repo_folder /home/element_symbolic_link
    - conda env create --file environment_ubuntu.yml

## Python Support

### Windows 10
#### Create virtual environment
    - cd python
    - conda env create --file environment_windows.yml
    
### Ubuntu
#### Create virtual environment
    - conda --version
        conda 4.10.3
    - cd python
    - conda env create --file environment_ubuntu.yml

### Misc

#### FAQ
    - pip install slow in China, use -i to select index source in China
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple your_package
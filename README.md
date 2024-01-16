# ELEMENT - CCTV Project

Sample code for users

## Setup

### OS Ubuntu
#### Create symbolic link to your repo folder
    - sudo ln -s /home/your_repo_folder /home/element_cctv_symbolic_link

#### Create virtual environment
```
# you need to install conda with version > 4.10.3

cd python
conda env create --file environment_ubuntu.yml

```


#### How to run QA script to make all the plots
```

cd python
./shell_scripts/run_qa.sh --batch

```

#### FAQ
    - pip install slow in China, use -i to select index source in China
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple your_package
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
conda activate element_ubuntu
cd python
./shell_scripts/run_qa.sh --batch --data_path your_path_to_the_processed_data

Example A:

conda activate element_ubuntu
cd python
./shell_scripts/run_qa.sh --batch --data_path /home/element_cctv_symbolic_link/data/

this will not render or pop out any windows, but dump out PNG files such as:



Example B:

conda activate element_ubuntu
cd python
./shell_scripts/run_qa.sh --data_path /home/element_cctv_symbolic_link/data/

this will pop out windows to render the plots, figures, you should run this on your
local computer or ssh to a machine with X-window enabled


```

#### FAQ
    - pip install slow in China, use -i to select index source in China
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple your_package
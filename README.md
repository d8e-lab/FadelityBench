# FadelityBench
## Environment
```bash
# create conda env with cuda
conda create -n fadelity python=3.10 cuda-toolkit=12.8 -c nvidia && conda activate fadelity
# install torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install transformers evaluate datasets gdown
cd bleurt && pip install .
# delete version in BARTScore/requirements.txt
cd ../BARTScore && pip install -r requirements.txt
```
## Models for Evaluation
download gdown using `pip install gdown`
```bash
# Barkscore model fine-tuned on ParaBank2
gdown 1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m
# BLEURT-20 for bleurt metric
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
```

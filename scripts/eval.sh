export API_KEY=""
export API_BASE_URL=""
# export http_proxy=127.0.0.1:7890
# export https_proxy=127.0.0.1:7890
datasets=(code.txt cot.txt fin.txt law.txt literature.txt math.txt medicine.txt science.txt sharegpt.txt sum.txt translate.txt)

BERT_MODEL="microsoft/deberta-xlarge-mnli" #"roberta-large"
BLEURT_MODEL="evaluation/models/BLEURT-20"
DATASET_BASE=data/pred/runs/
MODEL=llama3.1-8b-instruct
COMPRESS=topk
for dataset in "${datasets[@]}"; do
    echo ${dataset}
    DATA_PATH=${DATASET_BASE}${MODEL}
    python evaluation/main.py \
        --bert_model $BERT_MODEL \
        --bleurt_model $BLEURT_MODEL \
        --parabank \
        --predict_path ${DATA_PATH}-${COMPRESS}/${dataset} \
        --reference_path ${DATA_PATH}/${dataset}
done
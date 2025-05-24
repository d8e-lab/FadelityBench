export API_KEY=""
export API_BASE_URL=""
# export http_proxy=127.0.0.1:7890
# export https_proxy=127.0.0.1:7890

BERT_MODEL="microsoft/deberta-xlarge-mnli" #"roberta-large"
BLEURT_MODEL="evaluation/models/BLEURT-20"
predict_path="data/tst_pred.pkl"
reference_path="data/tst_ref.pkl"

python evaluation/main.py \
    --bert_model $BERT_MODEL \
    --bleurt_model $BLEURT_MODEL \
    --parabank \
    --predict_path $predict_path \
    --reference_path $reference_path \
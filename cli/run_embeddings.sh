# /bin/bash

DATA_ROOT=""
PARAMS_FOLDER=""
OUTPUT_PATH=""

for DIM in 128 256 512 768
do
    for MODALITY in "rnaseq" "mirna" "rppa" "dnameth"
    do 
    train-embeddings \
    --model_name AE_1layer_noactivation \
    --modality ${MODALITY} \
    --latent_dim ${DIM} \
    --data_path ${DATA_ROOT}/pancan_omics \
    --parameters_file ${PARAMS_FOLDER}/params_${MODALITY}.json \
    --output_path ${OUTPUT_PATH}
    done 
done 
DISABLE_TQDM=True

filename_list=(
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point58-3point09-direct-20250821"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-2point9-20250811"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-3point18-20250814"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point4-20250811"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point54-20250819"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-20250727"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point3-20250818"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point4-20250816-new"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point85-20250818"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point4-20250818-4node-mi300"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point57-20250821-4node"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-3point35-20250818"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point3-20250819"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point4-20250806"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point86-20250817"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-random-data-20250814"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-20250811-32-sigma3"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-20250814"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point075-4point0-3point62-3point13-20250826"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point075-4point0-3point62-3point13-direct-20250826"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point1-3point95-3point48-3point-direct-20250821"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-20250825"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-direct-20250821"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point58-3point09-20250823"
)

for file_name in "${filename_list[@]}"; 
do
    cp config.json /home/v-hongyihe/blob/${file_name}/ckpt-globalstep48000/
    lm_eval --model hf \
        --model_args "pretrained=/home/v-hongyihe/blob/${file_name}/ckpt-globalstep48000/,tokenizer=/home/v-hongyihe/blob/openpai/hongyi_he/Llama-3-8B-tokenizer" \
        --tasks mmlu_continuation \
        --device all \
        --output_path /home/v-hongyihe/blob/${file_name}/ckpt-globalstep48000/eval_results_append_mmlu_continuation.json \
        --trust_remote_code \

    lm_eval --model hf \
        --model_args "pretrained=/home/v-hongyihe/blob/${file_name}/ckpt-globalstep48000/,tokenizer=/home/v-hongyihe/blob/openpai/hongyi_he/Llama-3-8B-tokenizer" \
        --tasks mmlu_generative \
        --device all \
        --output_path /home/v-hongyihe/blob/${file_name}/ckpt-globalstep48000/eval_results_append_mmlu_generative.json \
        --trust_remote_code \

done
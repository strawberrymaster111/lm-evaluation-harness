DISABLE_TQDM=True

filename_list=(
    "hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point85-20250818"
    "hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-3point35-20250818"
    "hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point86-20250817"
    "hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-20250727/"
)

for file_name in "${filename_list[@]}"; 
do
    cp config.json /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep48000/
    lm_eval --model hf \
        --model_args "pretrained=/mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep48000/,tokenizer=/mnt/blob-openpai-xiaoliuinterns-out/hongyi_he/Llama-3-8B-tokenizer" \
        --tasks ifeval,xsum \
        --device all \
        --output_path /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep48000/eval_results_0127.json \
        --trust_remote_code \

    # lm_eval --model hf \
    #     --model_args "pretrained=/mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep48000/,tokenizer=/mnt/blob-openpai-xiaoliuinterns-out/openpai/hongyi_he/Llama-3-8B-tokenizer" \
    #     --tasks  \
    #     --device all \
    #     --output_path /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep48000/eval_results_0127.json \
    #     --trust_remote_code \

done
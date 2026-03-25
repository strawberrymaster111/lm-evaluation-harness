DISABLE_TQDM=True

filename_list=(
    # "hongyi_he/emb_pretrain/saved/nem-llama-400M-PPL725B-1115-2node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-400M-PPL100B-1115"
    "hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-PCaverage-700Bsample-1115-2node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-PCaverage-200Btop-1115"
    "hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-PCaverage-100Btop-1115"
    "hongyi_he/emb_pretrain/saved/nem-llama-400M-Nemotron-CC-HQ-1115-2node"
    "hongyi_he/emb_pretrain/saved/nem-llama-400M-Nemetron-CC-all-1115-2node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-joint-1115"
    # "hongyi_he/emb_pretrain/saved/nem-llama-400M-200B-joint-1115"
    "hongyi_he/emb_pretrain/saved/nem-llama-400M-120B-joint-1115"
    "hongyi_he/emb_pretrain/saved/nem-llama-400M-DSIR173B-1115"
)

for file_name in "${filename_list[@]}"; 
do
    for step in 48000
    do
        cp llama3_400M_config.json /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/config.json
        lm_eval --model hf \
            --model_args "pretrained=/mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/,tokenizer=/mnt/blob-openpai-xiaoliuinterns-out/hongyi_he/Llama-3-8B-tokenizer" \
            --tasks arc_easy,arc_challenge,hellaswag,sciq,piqa \
            --device all \
            --output_path /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/eval_results_append.json \
            --trust_remote_code \
            --log_samples \
            
    done

done
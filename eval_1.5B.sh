DISABLE_TQDM=True


    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-725B-20250918"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-100B-20250918"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-DSIR-173B-20250920"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-3point18-20250814"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point88-700B-0911"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-all-20250902-new"

filename_list=(
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-direct-20250821"
    "openpai/hongyi_he/emb_pretrain/saved/llama-score-top-data-4point02-3point85-3point35-2point86-direct-20250903"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotrodn-HQ-20250910-seed42-new"
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point1-3point95-3point48-3point-direct-20250821"
)

for file_name in "${filename_list[@]}"; 
do
    for step in 48000 96000
    do
        cp config.json /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/
        lm_eval --model hf \
            --model_args "pretrained=/mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/,tokenizer=/mnt/blob-openpai-xiaoliuinterns-out/hongyi_he/Llama-3-8B-tokenizer" \
            --tasks arc_easy,arc_challenge,hellaswag,sciq,piqa,humaneval,gsm8k \
            --device all \
            --output_path /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/eval_results_rebuttal.json \
            --trust_remote_code \
            --log_samples \
            
    done

done
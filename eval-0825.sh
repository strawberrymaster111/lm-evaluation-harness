DISABLE_TQDM=True

filename_list=(
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-direct-20250821"
    
)

for file_name in "${filename_list[@]}"; 
do
    for step in $(seq 1000 1000 48000)
    do
        cp config.json /mnt/${file_name}/ckpt-globalstep${step}/
        lm_eval --model hf \
            --model_args "pretrained=/mnt/${file_name}/ckpt-globalstep${step}/,tokenizer=/mnt/blob-openpai-xiaoliuinterns-out/hongyi_he/Llama-3-8B-tokenizer" \
            --tasks sciq,winogrande,race,openbookqa,arc_easy,arc_challenge,hellaswag,social_iqa,commonsense_qa,piqa,mmlu \
            --device all \
            --output_path /mnt/${file_name}/ckpt-globalstep${step}/eval_results_full_remote.json \
            --trust_remote_code \
            
    done

done
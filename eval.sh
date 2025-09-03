DISABLE_TQDM=True

filename_list=(
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-2point9-20250811"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-3point18-20250814"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point4-20250811"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point54-20250819"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-20250727"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point3-20250818"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point4-20250816-new"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point85-20250818"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point4-20250818-4node-mi300"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point57-20250821-4node"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-3point35-20250818"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point3-20250819"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point4-20250806"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point86-20250817"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-random-data-20250814"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-20250811-32-sigma3"
    "blob-openpai-xiaoliuinterns-out/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-20250814"
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
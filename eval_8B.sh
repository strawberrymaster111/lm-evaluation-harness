DISABLE_TQDM=True

filename_list=(
    #     "hongyi_he/emb_pretrain/saved/nem-llama-8b-400B-PCaverage-100Btop-1115-4node"
    #         "hongyi_he/emb_pretrain/saved/nem-llama-8b-200B-PCaverage-700Bsample-1115"
    # "hongyi_he/emb_pretrain/saved/nem-llama-8b-400B-PCaverage-100Btop-1115-4node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-8b-100B-nemotron-hq-1115-4node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-8b-400B-joint200B-1115-4node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-8b-400B-joint100B-1115-4node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-8b-100B-DSIR-1115-3node"
    "hongyi_he/emb_pretrain/working/nem-llama-8b-100B-PPL-sample-1115-4node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-8b-100B-PPL-sample-1115-4node"
)

    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-8b-200B-PCaverage-700Bsample-1115"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-8b-400B-PCaverage-100Btop-1115-4node"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-8b-100B-nemotron-hq-1115-4node"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-8b-400B-joint200B-1115-4node"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-8b-400B-joint100B-1115-4node"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-8b-100B-DSIR-1115-3node"

for file_name in "${filename_list[@]}"; 
do
    for step in 12000
    do
        cp llama3_8b_config.json /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/config.json
        lm_eval --model hf \
            --model_args "pretrained=/mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/,tokenizer=/mnt/blob-openpai-xiaoliuinterns-out/hongyi_he/Llama-3-8B-tokenizer" \
            --tasks arc_easy,arc_challenge,hellaswag,sciq,piqa \
            --device all \
            --output_path /mnt/blob-openpai-xiaoliuinterns-out/${file_name}/ckpt-globalstep${step}/eval_results_rebuttal.json \
            --trust_remote_code \
            
    done

done
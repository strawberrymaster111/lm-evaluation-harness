DISABLE_TQDM=True

lm_eval --model hf \
    --model_args "pretrained=/home/v-hongyihe/blob/openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-2point9-20250811/ckpt-globalstep48000/,tokenizer=/home/v-hongyihe/blob/openpai/hongyi_he/Llama-3-8B-tokenizer" \
    --tasks mmlu \
    --device cuda:0 \
    --output_path /home/v-hongyihe/blob/openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-2point9-20250811/ckpt-globalstep48000/eval_results_append.json \
    --trust_remote_code \


    # ,sciq,winogrande,race,openbookqa,arc_easy,arc_challenge,hellaswag,social_iqa,winogrande,race
    
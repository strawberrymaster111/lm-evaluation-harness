DISABLE_TQDM=True

# 模型文件所在的公共前缀目录
base_dir="hongyi_he/emb_pretrain/saved"

# checkpoint 目录名
ckpt="ckpt-globalstep48000"

# 挂载根目录
mount_root="/mnt/blob-openpai-xiaoliuinterns-out"

# ===== 模型目录列表 =====
# 直接粘贴目录名，每行一个：不需要引号、不需要逗号；
# 空行和以 # 开头的注释行会被自动忽略，行首尾多余空白会被去掉。
read -r -d '' model_text <<'MODELS'
nem-llama-15b-4096-hhy-score-top-data-50B-guard
MODELS

# 把文本块解析成数组：忽略空行/注释，去掉首尾空白，不需要引号
model_list=()
while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"   # 去左侧空白
    line="${line%"${line##*[![:space:]]}"}"   # 去右侧空白
    [ -z "${line}" ] && continue              # 跳过空行
    [ "${line:0:1}" = "#" ] && continue       # 跳过注释
    model_list+=("${line}")
done <<< "${model_text}"

# 多节点环境信息：优先用 PAI 变量，其次用已有环境变量，最后回退到单节点默认值
NODE_RANK=${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${NODE_RANK:-0}}
NUM_NODES=${PAI_TASK_ROLE_TASK_COUNT_worker:-${NUM_NODES:-1}}

# 防御：NUM_NODES 非法（空/0/非数字）时按单节点处理，避免取模除零
case "${NUM_NODES}" in
    ""|*[!0-9]*|0) NUM_NODES=1 ;;
esac
case "${NODE_RANK}" in
    ""|*[!0-9]*) NODE_RANK=0 ;;
esac
export NODE_RANK NUM_NODES

echo "NODE_RANK=${NODE_RANK}, NUM_NODES=${NUM_NODES}, total_models=${#model_list[@]}"

# 统计计数
total=0
success=0
failed_models=()

for idx in "${!model_list[@]}";
do
    # 按 node 数量对模型做取模均分，每个 node 只处理属于自己的部分
    if [ $(( idx % NUM_NODES )) -ne "${NODE_RANK}" ]; then
        continue
    fi

    model_name="${model_list[$idx]}"
    ckpt_dir="${mount_root}/${base_dir}/${model_name}/${ckpt}"
    total=$(( total + 1 ))
    echo "========================================================"
    echo "[NODE ${NODE_RANK}] evaluating (${total}): ${model_name}"

    # 校验 checkpoint 目录是否存在
    if [ ! -d "${ckpt_dir}" ]; then
        echo "[ERROR] checkpoint 目录不存在，跳过: ${ckpt_dir}"
        failed_models+=("${model_name} (目录不存在)")
        continue
    fi

    # 拷贝配置文件（失败也仅告警，不中断）
    if ! cp config.json "${ckpt_dir}/"; then
        echo "[WARN] 拷贝 config.json 失败: ${ckpt_dir}"
    fi

    # ---- 第 1 步：似然任务组（0-shot 默认） ----
    ok_main=1
    if ! lm_eval --model hf \
        --model_args "pretrained=${ckpt_dir}/,tokenizer=${mount_root}/hongyi_he/Llama-3-8B-tokenizer" \
        --tasks sciq,winogrande,race,openbookqa,arc_easy,arc_challenge,hellaswag,social_iqa,commonsense_qa,piqa,boolq,copa,wikitext,pile_10k,c4,lambada_openai,pile_arxiv,pile_github,pile_pubmed-central,pile_stackexchange,pile_wikipedia,pile_freelaw \
        --device all \
        --batch_size auto \
        --output_path "${ckpt_dir}/eval_results_0711.json" \
        --trust_remote_code ; then
        ok_main=0
        echo "[ERROR] 似然任务组评测失败: ${model_name}"
    fi

    # ---- 汇总本模型结果 ----
    if [ "${ok_main}" -eq 1 ]; then
        echo "[OK] 评测成功: ${model_name}"
        success=$(( success + 1 ))
    else
        detail=""
        [ "${ok_main}" -eq 0 ] && detail="${detail}似然组失败 "
        failed_models+=("${model_name} (${detail})")
    fi
done

echo "========================================================"
echo "[NODE ${NODE_RANK}] 评测完成: 成功 ${success}/${total}"
if [ "${total}" -gt 0 ]; then
    rate=$(awk "BEGIN{printf \"%.2f\", ${success}/${total}*100}")
    echo "[NODE ${NODE_RANK}] 成功率: ${rate}%"
fi
if [ "${#failed_models[@]}" -gt 0 ]; then
    echo "[NODE ${NODE_RANK}] 失败列表:"
    for m in "${failed_models[@]}"; do
        echo "  - ${m}"
    done
fi

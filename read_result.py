import json
import os
import glob

def find_eval_result_files(directory_path):
    """
    寻找指定文件夹中带有 eval_results_append_remote 字样的 JSON 文件
    
    Args:
        directory_path (str): 要搜索的文件夹路径
        
    Returns:
        list: 找到的文件路径列表
    """
    # 构建搜索模式
    search_pattern = os.path.join(directory_path, 'eval_results*.json')
    print(f"Searching for files in: {search_pattern}")
    
    # 使用 glob 查找匹配的文件
    matching_files = glob.glob(search_pattern)
    
    return matching_files


def extract_metrics_without_mmlu_subitems(file_path):
    """
    提取所有测试集的指标值，但过滤掉 mmlu 的子项（如 mmlu_humanities）
    
    Args:
        file_path (str): JSON 文件路径
        
    Returns:
        dict: 包含过滤后指标值的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        filtered_metrics = {}
        
        if 'results' in data and isinstance(data['results'], dict):
            for test_name, test_data in data['results'].items():
                # 过滤掉 mmlu 的子项（以 mmlu_ 开头但不是 mmlu 本身）
                if test_name.startswith('mmlu_'):
                    if test_name != 'mmlu_generative' and test_name != 'mmlu_continuation':
                        continue  # 跳过 mmlu 子项


                if test_name not in ['arc_challenge', 'arc_easy', 'hellaswag', 'piqa', 'sciq']:
                    print(f"跳过非目标测试集: {test_name}")
                    continue

                test_metrics = {}
                test_metrics['name'] = test_name
                
                # 提取 acc 相关指标
                if 'acc,none' in test_data:
                    test_metrics['acc'] = test_data['acc,none']
                if 'acc_stderr,none' in test_data:
                    test_metrics['acc_stderr'] = test_data['acc_stderr,none']
                
                # 提取 acc_norm 相关指标
                if 'acc_norm,none' in test_data:
                    test_metrics['acc_norm'] = test_data['acc_norm,none']
                if 'acc_norm_stderr,none' in test_data:
                    test_metrics['acc_norm_stderr'] = test_data['acc_norm_stderr,none']

                if 'exact_match,get_response' in test_data:
                    test_metrics['exact_match'] = test_data['exact_match,get_response']
                
                # 添加别名信息
                if 'alias' in test_data:
                    test_metrics['alias'] = test_data['alias']
                
                if test_metrics:  # 只有当有指标时才添加
                    filtered_metrics[test_name] = test_metrics
        
        return filtered_metrics
        
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return {}
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的 JSON")
        return {}
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return {}

def print_filtered_metrics_table(metrics_data):
    """
    以表格形式打印过滤后的指标数据
    """
    if not metrics_data:
        print("没有找到指标数据")
        return
    
    print("=" * 50)
    print(f"{'测试集':<18}{'acc/acc_norm':<10}")
    print("=" * 50)
    data_output = {}
    for test_name, metrics in metrics_data.items():
        acc = metrics.get('acc', 'N/A')
        acc_norm = metrics.get('acc_norm', 'N/A')
        acc_stderr = metrics.get('acc_stderr', 'N/A')
        acc_norm_stderr = metrics.get('acc_norm_stderr', 'N/A')
        exact_match = metrics.get('exact_match', 'N/A')
        # print(f"{test_name:<18}")
        data_output['test_name'] = test_name
        # 格式化输出
        if isinstance(acc_norm, float):
            # print(f"{acc:.4f}")
            data_output['acc'] = acc_norm
        elif isinstance(acc, float):
            # print(f"{acc_norm:.4f}")
            data_output['acc'] = acc

        if isinstance(exact_match, float):
            # print(f"{exact_match:.4f}")
            data_output['exact_match'] = exact_match
        if isinstance(acc_norm_stderr, float):
            # print(f"±{acc_stderr:.4f}")
            data_output['acc_stderr'] = acc_norm_stderr
        elif isinstance(acc_stderr, float):
            # print(f"±{acc_norm_stderr:.4f}")
            data_output['acc_stderr'] = acc_stderr

        print(f"{test_name:<18}{data_output['acc']:.4f} ±{data_output['acc_stderr']:.4f}")
        print("-" * 50)


# 使用示例
if __name__ == "__main__":
    # 替换为你的 JSON 文件路径
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point57-20250821-2node"
    # "hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point58-3point09-direct-20250821"

    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point58-3point09-20250823"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-20250825"
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point075-4point0-3point62-3point13-20250826"

    # path_list = ["openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point58-3point09-direct-20250821",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-2point9-20250811",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-3point18-20250814",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point4-20250811",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point54-20250819",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-20250727",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point3-20250818",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point4-20250816-new",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point85-20250818",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point4-20250818-4node-mi300",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point57-20250821-4node",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-3point35-20250818",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point3-20250819",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point4-20250806",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point86-20250817",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-random-data-20250814",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-20250811-32-sigma3",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-20250814",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point075-4point0-3point62-3point13-20250826",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point075-4point0-3point62-3point13-direct-20250826",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point1-3point95-3point48-3point-direct-20250821",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-20250825",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-direct-20250821",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point58-3point09-20250823"]

    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-all-20250902-new",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point1-3point95-3point48-3point-direct-20250903-seed32",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point1-3point95-3point48-3point-direct-20250903-seed32",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-direct-20250903-seed32",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point02-3point85-3point35-2point86-direct-20250903",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point22-4point19-3point53-3point0-direct-20250830",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-HQ-20250902-new",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-top-data-4point22-4point05-3point53-3point0-direct",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-all-20250902-new",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-20250903-seed32",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-top-data-20B-0906-direct",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-all-20250908-new-seed21",
    # "openpai/hongyi_he/emb_pretrain/saved/top-data-40B-PC1-3-0908",
    # "openpai/hongyi_he/emb_pretrain/saved/top-data-24B-PC1-5-0908",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point99-400B-0911",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-top-data-20B-0911-direct-seed42",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-3point08-200B-0911",
    # "openpai/hongyi_he/emb_pretrain/saved/top-data-24B-PC1-5-0911-seed42",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-HQ-20250910-seed42-new",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-top-data-30B-0906-direct",
#     path_list = [
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point80-900B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point84-800B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point95-500B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-3point04-300B-0911",
#         "openpai/hongyi_he/emb_pretrain/saved/top-data-60B-PC1-2-0908-new",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point91-600B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point88-700B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-3point08-200B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-joint-150B-20250920",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-100B-20250918",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-DSIR-173B-20250920",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-725B-20250918",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-joint-200B-20250920"
# ]

#     path_list = [
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-HQ-20250910-seed42-new",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point88-700B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-all-20250908-new-seed21",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-3point18-20250814",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-100B-20250918",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-DSIR-173B-20250920",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-725B-20250918",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-direct-20250821"
# ]

#     path_list = [
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point80-900B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point84-800B-0911",
#         "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point88-700B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point91-600B-0911",
#         "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point95-500B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point99-400B-0911",
#         "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-3point04-300B-0911",
#     "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-3point08-200B-0911",
#         "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-3point18-20250814",
# ]

    path_list = [
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-PPL725B-1115-2node",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-PPL100B-1115",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-PCaverage-700Bsample-1115",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-PCaverage-200Btop-1115",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-PCaverage-100Btop-1115",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-Nemotron-CC-HQ-1115-2node",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-Nemetron-CC-all-1115-2node",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-400B-joint-1115",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-200B-joint-1115",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-120B-joint-1115",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-400M-DSIR173B-1115"
    ]

    path_list = [
            "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-20250727",
                "openpai/hongyi_he/emb_pretrain/saved/top-data-60B-PC1-2-0908-new",
            "openpai/hongyi_he/emb_pretrain/saved/top-data-40B-PC1-3-0908",
                "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-20250903-seed32",
    "openpai/hongyi_he/emb_pretrain/saved/top-data-24B-PC1-5-0908",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-top-data-20B-0906-direct",]

    path_list = [
            "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-725B-20250918",
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-ppl-100B-20250918",
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-DSIR-173B-20250920",
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-all-3point18-20250814",
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-average-2point88-700B-0911",
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-all-20250902-new",
        "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point16-4point0-3point56-3point06-direct-20250821",
    "openpai/hongyi_he/emb_pretrain/saved/llama-score-top-data-4point02-3point85-3point35-2point86-direct-20250903",
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-nemotron-HQ-20250910-seed42-new",
    "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-top-data-4point1-3point95-3point48-3point-direct-20250821"
    ]


    # path_list = [
    #         "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point4-20250811",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-20250727",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point3-20250818",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc2-3point85-20250818",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-2point4-20250818-4node-mi300",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc3-3point35-20250818",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point3-20250819",
    # "openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc4-2point86-20250817",]



    for path in path_list:
        directory_path = f"/home/v-hongyihe/blob/{path}/ckpt-globalstep96000"

        # 查找匹配的文件
        eval_files = find_eval_result_files(directory_path)
        
        if not eval_files:
            print(f"在文件夹 {directory_path} 中没有找到带有 'eval_results_append' 字样的 JSON 文件")
            print("请检查：")
            print("1. 文件夹路径是否正确")
            print("2. 文件名是否包含 'eval_results_append_remote'")
            print("3. 文件扩展名是否为 .json")
        else:
            print(f"找到 {len(eval_files)} 个匹配的文件:")
            for i, file_path in enumerate(eval_files, 1):
                print(f"{i}. {os.path.basename(file_path)}")
            
            # 处理每个找到的文件
            for file_path in eval_files:
                file_name = os.path.basename(file_path)
                print(f"\n{'='*60}")
                print(f"处理文件: {file_name}")
                print(f"{'='*60}")
        
                # 提取过滤后的指标数据（排除 mmlu 子项）
                filtered_metrics = extract_metrics_without_mmlu_subitems(file_path)
                
                print(path)

                # 打印表格
                print("过滤后的测试集指标（排除 mmlu 子项）：")
                print_filtered_metrics_table(filtered_metrics)
                
                # 显示包含的测试集列表
                print(f"\n包含的测试集: {list(filtered_metrics.keys())}")
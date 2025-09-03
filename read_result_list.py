import json
import os
import glob
import numpy as np



def find_eval_result_files(directory_path):
    """
    寻找指定文件夹中带有 eval_results_append_remote 字样的 JSON 文件
    
    Args:
        directory_path (str): 要搜索的文件夹路径
        
    Returns:
        list: 找到的文件路径列表
    """
    # 构建搜索模式
    search_pattern = os.path.join(directory_path, '*eval_results_full_remote*.json')
    
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
                if test_name.startswith('mmlu_') and test_name != 'mmlu':
                    continue  # 跳过 mmlu 子项
                
                test_metrics = {}
                
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
    
    # print("=" * 50)
    # print(f"{'测试集':<18}{'acc/acc_norm':<10}")
    # print("=" * 50)
    list = []
    test_name_list = []
    for test_name, metrics in metrics_data.items():
        acc = metrics.get('acc', 'N/A')
        acc_norm = metrics.get('acc_norm', 'N/A')
        
        # 格式化输出
        if isinstance(acc, float):
            # print(f"{acc:.4f}")
            list.append(acc)
            test_name_list.append(test_name+' acc')
        if isinstance(acc_norm, float):
            # print(f"{acc_norm:.4f}")
            list.append(acc_norm)
            test_name_list.append(test_name+' acc_norm')
    
    return list, test_name_list


def print_structured_output(results_dict):
    """
    结构化输出结果字典
    """
    if not results_dict:
        print("没有可输出的数据")
        return
    
    result_list = []
    for file_name, file_data in results_dict.items():
        new_list = []
        new_list.append(file_name)
        for i in range(len(file_data)):
            new_list.append(file_data[i])
        result_list.append(new_list)
    
    result_list = np.array(result_list)
    np.set_printoptions(precision=4, suppress=True)
    print(result_list)

    return result_list

def plot_results(result_list, test_name_list):
    import matplotlib.pyplot as plt

    result_array = np.array(result_list)
    steps = result_array[:, 0]
    values = result_array[:, 1:].astype(float)

    plt.figure(figsize=(12, 6))
    
    for i in range(values.shape[1]):
        if i < 10:
            plt.plot(steps, values[:, i], marker='o', label=test_name_list[i])
    
    plt.title('Metrics over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Metric Values')
    plt.legend()
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("metrics_over_steps_part1.png")

    plt.figure(figsize=(12, 6))
    
    for i in range(values.shape[1]):
        if i >= 10:
            plt.plot(steps, values[:, i], marker='o', label=test_name_list[i])
    
    plt.title('Metrics over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Metric Values')
    plt.legend()
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("metrics_over_steps_part2.png")



# 使用示例
if __name__ == "__main__":
    # 替换为你的 JSON 文件路径
    result_dict = {}

    for step in range(1000, 48000, 1000):
        directory_path = f"/home/v-hongyihe/blob/openpai/hongyi_he/emb_pretrain/saved/nem-llama-15b-4096-hhy-score-pc1-3point4-20250811/ckpt-globalstep{step}"

        # 查找匹配的文件
        eval_files = find_eval_result_files(directory_path)
        
        if not eval_files:
            print(f"在文件夹 {directory_path} 中没有找到带有 'eval_results_full_remote' 字样的 JSON 文件")
            print("请检查：")
            print("1. 文件夹路径是否正确")
            print("2. 文件名是否包含 'eval_results_full_remote'")
            print("3. 文件扩展名是否为 .json")
        else:

            # 处理每个找到的文件
            for file_path in eval_files:
                file_name = os.path.basename(file_path)
                # print(f"\n{'='*60}")
                # print(f"处理文件: {file_name}")
                # print(f"{'='*60}")
        
            # 提取过滤后的指标数据（排除 mmlu 子项）
            filtered_metrics = extract_metrics_without_mmlu_subitems(file_path)
            
            # 打印表格
            # print("过滤后的测试集指标（排除 mmlu 子项）：")
            list_tmp, test_name_list = print_filtered_metrics_table(filtered_metrics)
            result_dict[step] = list_tmp

            # 显示包含的测试集列表
            # print(f"\n包含的测试集: {list(filtered_metrics.keys())}")

    result_list = print_structured_output(result_dict)
    plot_results(result_list, test_name_list)
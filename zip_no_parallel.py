import zlib
import torch
from tqdm import tqdm
import time
import pandas as pd
from collections import defaultdict
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

# 计算压缩比
def get_compression_ratio(input_data):
    data_str = str(input_data).encode('utf-8')
    compressed_data = zlib.compress(data_str, level=9)
    compressed_ratio = len(data_str) / len(compressed_data)
    return compressed_ratio

# 从DataFrame中选择数据
def selec_data_from_corpus(
    anchor_data,
    processed_data_index,
    budget,
    selection_num,
    candidate_budget='all',
    turn_print=True,
    data_pool=None,
    global_information_redundancy_state=None,
):
    data_list = [data_pool.iloc[_] for _ in processed_data_index]
    selected_data = []
    selected_index = []

    if not turn_print:
        start_time = time.time()

    while True:
        if turn_print:
            start_time = time.time()

        if candidate_budget == 'all':
            # 计算冗余度
            group_information_redundancy_state = [
                get_compression_ratio(str(anchor_data + selected_data + [part])) for part in tqdm(data_list)
            ]
            print(group_information_redundancy_state)
            group_information_redundancy_state = torch.tensor(group_information_redundancy_state)
            group_information_redundancy_state[selected_index] = 1000000  # 已选择的索引设置为很大值
            _, min_index = torch.topk(group_information_redundancy_state, k=selection_num, largest=False)
            new_index = min_index.tolist()

            selected_instance_list = []
            for _ in new_index:
                selected_instance = data_list[_]
                selected_instance_list.append(selected_instance)
            selected_index.extend(new_index)
            selected_data.extend(selected_instance_list)
        else:
            # 使用全局冗余状态
            # 确保 cur_index 中的元素是整数
            cur_index = [int(idx) for idx in global_information_redundancy_state.tolist()]
            _, cur_index = torch.topk(global_information_redundancy_state, k=candidate_budget, largest=False)
            group_list = [data_pool.iloc[int(idx)] for idx in cur_index]
            
            # 使用 tqdm 显示进度条
            print("Stage 3: Calculating compression ratios...")
            group_information_redundancy_state = []
            for part in tqdm(group_list, desc="Calculating compression", unit="item"):
                group_information_redundancy_state.append(get_compression_ratio(str(anchor_data + selected_data + [part])))
            
            group_information_redundancy_state = torch.tensor(group_information_redundancy_state)
            global_information_redundancy_state[cur_index] = group_information_redundancy_state
            _, min_index = torch.topk(group_information_redundancy_state, k=selection_num, largest=False)
            new_index = cur_index[min_index].tolist()
            global_information_redundancy_state[new_index] = 1000000
            selected_instance_list = []
            for _ in new_index:
                selected_instance = data_pool.iloc[int(_) ]  # Make sure to convert to integer
                selected_instance_list.append(selected_instance)
            selected_index.extend(new_index)
            selected_data.extend(selected_instance_list)

        if turn_print:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Code execution time: {execution_time} seconds")

        cur_len = len(selected_data)
        if cur_len >= budget:
            selected_global_index = [processed_data_index[_] for _ in selected_index]
            if not turn_print:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Code execution time: {execution_time} seconds")
            return selected_global_index, selected_data

# ZIP选择函数
def ZIP_select(data_pool, save_path, budget, k1=10000, k2=200, k3=100):
    # 移除并行代码部分
    # pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), n_jobs))
    
    # 初始化数据冗余状态
    global_information_redundancy_state = [
        get_compression_ratio(str(part)) for _, part in data_pool.iterrows()
    ]
    global_information_redundancy_state = torch.tensor(global_information_redundancy_state)

    final_selected_data = []
    cur_data_index = list(range(len(data_pool)))
    
    while len(final_selected_data) < budget:
        print('stage 1 & stage 2')
        second_stage_index, _ = selec_data_from_corpus(
            final_selected_data, cur_data_index, k2, k2, k1, turn_print=True,
            data_pool=data_pool,
            global_information_redundancy_state=global_information_redundancy_state
        )
        print('stage 3')
        third_stage_index, third_stage_data = selec_data_from_corpus(
            [], second_stage_index, k3, 1, k3, turn_print=False,
            data_pool=data_pool,
            global_information_redundancy_state=global_information_redundancy_state
        )
        cur_data_index = [_ for _ in cur_data_index if _ not in third_stage_index]
        final_selected_data.extend(third_stage_data)
        
        source_list = defaultdict(int)
        for _ in final_selected_data:
            source_list[_['source']] += 1  # 如果没有 'source' 列，可以修改为其他列
        print(f'selected {len(final_selected_data)}, including {source_list}')
    
    # 将最终选择的数据保存到文件
    final_selected_df = pd.DataFrame(final_selected_data)
    final_selected_df.to_json(save_path, orient='records', lines=True)

# 主函数
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data.csv', help='输入数据的CSV文件路径')
    parser.add_argument('--save_path', type=str, default='./zip_selected_data.json', help='保存选定数据的路径')
    parser.add_argument('--budget', type=int, default=1000, help='选定的实例数量')
    parser.add_argument('--k1', type=int, default=10000, help='阶段1的样本数')
    parser.add_argument('--k2', type=int, default=200, help='阶段2的样本数')
    parser.add_argument('--k3', type=int, default=100, help='阶段3的样本数')
    
    args = parser.parse_args()
    
    # 读取输入的CSV文件到DataFrame
    df = pd.read_csv(args.data_path, header=None, names=['x_1', 'x_2', 'y'])  # 根据需要指定列名
    
    # 运行ZIP_select函数
    ZIP_select(df, args.save_path, args.budget, args.k1, args.k2, args.k3)

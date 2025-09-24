import numpy as np
from sklearn.cluster import  KMeans
from sklearn.manifold import TSNE
import torch
import argparse
import json
import os
from tqdm import tqdm
import pandas as pd
from FlagEmbedding import FlagModel
import time
import submodlib
from submodlib.functions.facilityLocation import FacilityLocationFunction
from datasets import Dataset

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import yaml
import argparse

import torch.multiprocessing as mp
import subprocess

# yaml_path = 'script/config_CTA_WebTable_multi_device.yaml'

# yaml_path = 'script/config_RE_RE_vary_batch_size.yaml'

yaml_path = 'script/config_CTA_WebTable_vary_batch_size.yaml'
import os
import shutil

def remove_dir_if_exists(path: str):
    """
    删除目标文件夹（如果存在）。
    :param path: 目标路径
    """
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted folder: {path}")
    else:
        print(f"Folder does not exist: {path}")
def is_folder_empty(folder_path):
    """
    检查指定的文件夹是否为空文件夹。

    参数:
    folder_path (str): 要检查的文件夹的路径

    返回:
    bool: 如果文件夹为空，返回True；否则返回False。
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"指定的路径 {folder_path} 不存在，请检查输入的文件夹路径是否正确。")
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} 不是一个文件夹，请确保输入的是文件夹路径。")

    # 获取文件夹下的所有文件和子文件夹列表
    contents = os.listdir(folder_path)
    return len(contents) == 0

def z_score_normalize(sample_IF):
    """
    对sample_IF字典中各个'method'对应的值（键为index、值为float的字典）进行Z-score归一化。

    参数:
    sample_IF (dict): 包含多个'method'键的字典，每个'method'键对应的值为需要进行归一化处理的字典数据。

    返回:
    dict: 归一化后的字典，结构与输入的sample_IF一致，其中每个'method'键对应的值都已经完成Z-score归一化。
    """
    for method in sample_IF.keys():
        # 获取当前method对应需要归一化的值列表，保持原有顺序
        values_list = list(sample_IF[method].values())
        # 将列表转换为torch.Tensor
        tensor_value = torch.tensor(values_list).unsqueeze(1)  # 添加维度，变为二维张量

        # 计算均值和标准差
        mean_value = tensor_value.mean()
        std_value = tensor_value.std()

        # 进行Z-score归一化
        normalized_tensor = (tensor_value - mean_value) / std_value

        # 将归一化后的结果再转换回列表
        normalized_list = normalized_tensor.squeeze(1).tolist()

        # 更新原字典中当前method对应的值
        index_list = list(sample_IF[method].keys())
        normalized_dict = {}
        for index, normalized_value in zip(index_list, normalized_list):
            normalized_dict[index] = normalized_value

        sample_IF[method] = normalized_dict

    return sample_IF

def run_command(command):
    """
    执行给定的shell命令，并返回执行结果。
    """
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"命令 {e.cmd} 执行出错: {e.stderr.decode('utf-8')}")
        return None

def cluster_vectors(high_dim_vectors_cluster, indexes, batch_size):
    """
    根据给定的高维向量矩阵、索引以及期望的每个聚类元素个数进行聚类划分。

    参数:
    high_dim_vectors_cluster (numpy.ndarray): m * n的高维向量矩阵，m为元素个数，n为向量维度。
    indexes (numpy.ndarray): 对应高维向量矩阵中元素的索引列表，长度为k。
    batch_size (int): 期望每个聚类包含的元素个数。

    返回:
    tuple: 包含两个元素，第一个元素是聚类划分后的结果（字典形式，键为聚类编号，值为对应聚类包含的元素索引列表），
           第二个元素是覆盖率（float类型，表示已分配元素占总元素的比例）。
    """
    # 获取对应索引的向量
    selected_vectors = high_dim_vectors_cluster[indexes]

    # 计算这些向量之间的余弦相似度矩阵
    similarity_matrix = cosine_similarity(selected_vectors)

    # 使用层次聚类（AgglomerativeClustering）基于余弦相似度进行聚类划分
    k = len(indexes)
    clustering_model = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    clustering_model.fit(similarity_matrix)

    # 根据聚类结果分配所有的m个元素到对应的聚类中
    cluster_assignments = {i: [] for i in range(k)}
    for i in range(len(high_dim_vectors_cluster)):
        vector = high_dim_vectors_cluster[i].reshape(1, -1)
        similarities = cosine_similarity(vector, selected_vectors)[0]
        closest_cluster = np.argmax(similarities)
        cluster_assignments[closest_cluster].append(i)

    # 检查未分配的元素，并根据距离的就近原则分配到对应的聚类中
    all_indices = set(range(len(high_dim_vectors_cluster)))
    assigned_indices = set([index for sublist in cluster_assignments.values() for index in sublist])
    unassigned_indices = all_indices - assigned_indices
    for index in unassigned_indices:
        vector = high_dim_vectors_cluster[index].reshape(1, -1)
        distances = []
        for cluster_id in range(k):
            cluster_vectors = np.array([high_dim_vectors_cluster[i].reshape(1, -1) for i in cluster_assignments[cluster_id]])
            mean_cluster_vector = np.mean(cluster_vectors, axis=0)
            distance = cosine_similarity(vector, mean_cluster_vector)[0][0]
            distances.append(distance)
        closest_cluster = np.argmin(distances)
        cluster_assignments[closest_cluster].append(index)

    # 调整每个聚类中的元素个数尽量接近batch_size（这里简单处理，可根据实际优化）
    for cluster_id in range(k):
        cluster_indices = cluster_assignments[cluster_id]
        if len(cluster_indices) > batch_size:
            cluster_assignments[cluster_id] = cluster_indices[:batch_size]
        elif len(cluster_indices) < batch_size:
            while len(cluster_indices) < batch_size and unassigned_indices:
                # 从未分配元素中找距离当前聚类最近的补充进来
                index_to_add = None
                min_distance = float('inf')
                for unassigned_index in unassigned_indices:
                    unassigned_vector = high_dim_vectors_cluster[unassigned_index].reshape(1, -1)
                    mean_cluster_vector = np.mean([high_dim_vectors_cluster[i].reshape(1, -1) for i in cluster_indices], axis=0)
                    distance = cosine_similarity(unassigned_vector, mean_cluster_vector)[0][0]
                    if distance < min_distance:
                        min_distance = distance
                        index_to_add = unassigned_index
                if index_to_add is not None:
                    cluster_assignments[cluster_id].append(index_to_add)
                    unassigned_indices.remove(index_to_add)
                    cluster_indices = cluster_assignments[cluster_id]

    # 计算覆盖率
    coverage = len(assigned_indices) / len(high_dim_vectors_cluster)

    return cluster_assignments, coverage
def cosine_similarity_clustering(high_dim_vectors_cluster, indices, k, batch_size):
    """
    Perform k clustering based on cosine similarity, ensuring each cluster has batch_size elements.

    Parameters:
    - high_dim_vectors_cluster (np.ndarray): An m x n matrix containing m high-dimensional vectors.
    - indices (list): A list of k indices corresponding to m rows in the matrix.
    - k (int): Number of clusters.
    - batch_size (int): Desired number of elements per cluster.

    Returns:
    - clusters (list of lists): A list where each sublist contains the indices of the elements in a cluster.
    """
    m, n = high_dim_vectors_cluster.shape
    if len(indices) != k:
        raise ValueError("Number of provided indices must match the number of clusters (k).")

    # Step 1: Initialize cluster centers using the given indices
    cluster_centers = high_dim_vectors_cluster[indices]

    # Step 2: Compute cosine similarity between all elements and cluster centers
    similarity_matrix = cosine_similarity(high_dim_vectors_cluster, cluster_centers)

    # Step 3: Assign elements to clusters greedily (allow duplicates if needed)
    clusters = [[] for _ in range(k)]

    for _ in range(batch_size):  # Ensure each cluster has batch_size elements
        for cluster_idx in range(k):
            # Find the most similar element for the current cluster
            best_idx = -1
            best_similarity = -1
            for i in range(m):
                if similarity_matrix[i, cluster_idx] > best_similarity:
                    best_idx = i
                    best_similarity = similarity_matrix[i, cluster_idx]

            if best_idx != -1:
                clusters[cluster_idx].append(best_idx)

    # Step 4: Handle remaining elements by assigning to closest clusters
    remaining_elements = [i for i in range(m)]
    for i in remaining_elements:
        # Assign to the cluster with the highest similarity
        best_cluster = np.argmax(similarity_matrix[i])
        clusters[best_cluster].append(i)

    return clusters

def do_fla(X, number_all, number_select):
    start_time = time.time()

    Y = X
    obj = FacilityLocationFunction(n=number_all, mode="dense", data=Y, metric="cosine")
    
    greedyList = obj.maximize(budget=number_select, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    idx_list = [tuple_i[0] for tuple_i in greedyList]

    # print('FLA time used:',(time.time()-start_time),'(second)')
    return idx_list,greedyList

def create_folder_for_file(file_path):
    """
    给定一个文件的路径，判断路径中的文件夹是否存在，不存在则创建。

    参数:
    file_path (str): 文件的完整路径，例如 "A/B/C/df.csv"

    返回:
    None
    """
    # 获取文件所在目录的路径（去除文件名部分）
    folder_path = os.path.dirname(file_path)
    current_path = ""
    # 分割路径字符串为各个文件夹名组成的列表
    folders = folder_path.split(os.path.sep)
    for folder in folders:
        current_path = os.path.join(current_path, folder)
        if not os.path.exists(current_path):
            os.mkdir(current_path)

def do_clustering(high_dim_vectors,cluster_method='kmeans',kmeans_num_clusters=100): ## clustering 

    clustering_algorithm = cluster_method
    if clustering_algorithm == 'kmeans':
        clustering = KMeans(n_clusters=kmeans_num_clusters, random_state=0).fit(high_dim_vectors)
    
    return clustering

def do_reduce_dim(high_dim_vectors): ## draw t-sne
    # Perform t-SNE for visualization
    # if args.reduce_method == 'tsne':
    tsne = TSNE(n_components=2, random_state=0)
    low_dim_vectors = tsne.fit_transform(high_dim_vectors)
    return low_dim_vectors

def sample_middle_confidence_data(cluster_labels, confidences, n, low_th=25, up_th=75): ## 第一阶段draft model training的数据筛选模型,可以跳过
    num_clusters = len(np.unique(cluster_labels))

    # Get the indices for each cluster
    cluster_indices = {i: np.where(cluster_labels == i)[0] for i in range(num_clusters)}
    
    # Create a dictionary to store the indices of the middle level confidence samples
    middle_confidence_samples = {}

    for i in range(num_clusters):
        # Get the sorted indices for this cluster
        sorted_indices = cluster_indices[i]
        
        # If there are less than n samples in this class, just return all of them
        if len(sorted_indices) < n:
            middle_confidence_samples[i] = sorted_indices
            continue

        # Get the confidences for this cluster
        cluster_confidences = confidences[sorted_indices]
        lower_threshold = np.percentile(cluster_confidences, low_th)
        upper_threshold = np.percentile(cluster_confidences, up_th)

        # Get the indices of the samples within the middle level confidence range
        middle_indices = sorted_indices[(cluster_confidences >= lower_threshold) & (cluster_confidences <= upper_threshold)]
        
        # If there are less than n samples in the middle range, use all of them
        if len(middle_indices) < n:
            middle_confidence_samples[i] = middle_indices
        else:
            # Calculate step size for even sampling
            step_size = len(middle_indices) // n
            # Select evenly from the middle level confidence samples
            middle_confidence_samples[i] = middle_indices[::step_size][:n]

    return middle_confidence_samples

int_arg_list = ['cluster_num','sample_per_cluster','batch_size']

def load_yaml_args(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        args_dict = yaml.safe_load(file)
    print(args_dict)
    parser = argparse.ArgumentParser()
    for key, value in args_dict.items():
        if isinstance(value, bool):
            # 将arg3指定为存储布尔值的参数，action='store_true'表示如果参数出现则为True
            parser.add_argument(f'--{key}', action='store_true', default = value)
        elif key in int_arg_list:
            parser.add_argument(f'--{key}', type = int, default = int(value))
        else:
            parser.add_argument(f'--{key}', default = value)
    args = parser.parse_args()
    return args

def get_json_sample(middle_confidence_samples):
    
    json_samples = []
    for k in middle_confidence_samples.keys():
        ids_list = middle_confidence_samples[k].tolist()
        # for id_i in ids_list:
            # ori_sample = json_data[id_i]
        json_samples.extend(ids_list)
    

    return json_samples

def CTA_Embed_Text(df): ## 返回3列分别是text_1,text_2,label
    SimTab = df
    context_list = []
    target_list = []
    label_list = []
    for i in range(len(SimTab)):
        text = SimTab.iloc[i,0]
        label = list(eval(SimTab.iloc[i,2]).values())[0]
        label_list.append(label) ## add label
        context = text.split('Table 1:\n\n')[1].split('\n\nReference tables:')[0].split('\n')
        context_list.append(str(context)) ## add context
        
        target = text.split('\n\nColumn: ')[1].split('\n\nOptions:')[0]
        related_context = []
        for candidate in context:
            try:
                dict_output = eval(candidate)
                related_context.append(dict_output[target])
            except:
                print('fail')
                print(dict_output.keys(),target)
        target_list.append(str({target:related_context}))
    return context_list,target_list,label_list
def RE_Embed_Text(df): ## 返回3列分别是text_1,text_2,label
    SimTab = df
    context_list = []
    target_list = []
    label_list = []
    for i in range(len(SimTab)):
        text = SimTab.iloc[i,0]
        label = list(eval(SimTab.iloc[i,2]).values())[0]
        label_list.append(label) ## add label
        context = text.split('\n\nTable1:')[1].split('\n\nRelation Option: \n\n')[0].replace('<table_title>','<table_title> ').replace('<header>',' <header> ').replace('|',' | ')
        context_list.append(str(context)) ## add context
        target = text.split('<header>')[1].split('\n\nRelation Option: \n\n')[0]
        target_list.append(str(target.replace('|',' | ')))
    return context_list,target_list,label_list
def ER_Embed_Text(df): ## 返回3列分别是text_1,text_2,label
    SimTab = df
    context_list = []
    target_list = []
    label_list = []
    for i in range(len(SimTab)):
        text = SimTab.iloc[i,0]
        label = list(eval(SimTab.iloc[i,2]).values())[0]
        label_list.append(label) ## add label
        context = text.split('Output format example:{"Output": ""}\n\nEntity 1:')[1].split('\n\nEntity 2:')[0].replace('\\n', '').replace('\n', '').replace('\\', '')
        context_list.append(context) ## add context
        
        target = text.split('\n\nEntity 2:')[1].split('\n\nTake these examples as reference:')[0].replace('\\n', '').replace('\n', '').replace('\\', '')
        target_list.append(target)
    return context_list,target_list,label_list
# parser_yaml = argparse.ArgumentParser()
# parser_yaml.add_argument('--yaml_path', type = str, default = 'config.yaml')
# args_yaml = parser_yaml.parse_args()

def DC_Embed_Text(df): ## 返回3列分别是text_1,text_2,label
    SimTab = df
    context_list = []
    target_list = []
    label_list = []
    for i in tqdm(range(len(SimTab))):
        text = SimTab.iloc[i,0]
        label = SimTab.iloc[i,2]
        label_list.append(label) ## add label
        if text.__contains__('You are an expert in Cleaning Hospital Dataset.'):
            context = text.split('Entity 1:\n\n')[1].split('\n\nTake these rows as reference:')[0]
            context_list.append(context) ## add context
            
            context_dict = eval(context)
            
            target = text.split('Output Format Example:\n\n')[1].split('\n\nEntity 1:')[0]
            
            target_attr = list(eval(target).keys())[0]
            
            related_context = context_dict[target_attr]
            target_list.append(str({target_attr:related_context}))
        elif text.__contains__('You are an expert in Cleaning Beers Dataset.'):
            context = text.split('Entity 1:\n\n')[1].split('\n\nThe input')[0]
            context_list.append(context) ## add context
            
            context_dict = eval(context)
            
            target = text.split('Output Format Example:\n\n')[1].split('\n\nEntity 1:')[0]
            
            target_attr = list(eval(target).keys())[0]
            
            related_context = context_dict[target_attr]
            target_list.append(str({target_attr:related_context}))
        elif text.__contains__('You are an expert in Cleaning Rayyan Dataset.'):
            context = text.split('Entity 1:\n\n')[1].split('\n\nThe input')[0]
            context_list.append(context) ## add context
            
            context_dict = eval(context)
            
            target = text.split('Output Format Example:\n\n')[1].split('\n\nEntity 1:')[0]
            
            target_attr = list(eval(target).keys())[0]
            
            related_context = context_dict[target_attr]
            target_list.append(str({target_attr:related_context}))
    return context_list,target_list,label_list
def DI_Embed_Text(df): ## 返回3列分别是text_1,text_2,label
    SimTab = df
    context_list = []
    target_list = []
    label_list = []
    for i in tqdm(range(len(SimTab))):
        text = SimTab.iloc[i,0]
        label = SimTab.iloc[i,2]
        label_list.append(label) ## add label
        context = text.split('\n\nEntity 1:')[1].split('\n\nTake these examples as reference:')[0]
        context_list.append(context) ## add context
        
        # context_dict = eval(context)
        
        target = text.split('\n\nOutput format example:')[1].split('\n\nEntity 1:')[0]
        
        # target_attr = list(eval(target).keys())[0]
        
        # related_context = context_dict[target_attr]
        target_list.append(target)

    return context_list,target_list,label_list

def SM_Embed_Text(df): ## 返回3列分别是text_1,text_2,label
    SimTab = df
    context_list = []
    target_list = []
    label_list = []
    for i in range(len(SimTab)):
        text = SimTab.iloc[i,0]
        label = list(eval(SimTab.iloc[i,2]).values())[0]
        label_list.append(label) ## add label
        context = text.split('Output format example:{"Output": ""}\n\nColumn 1:')[1].split('\n\nColumn 2:')[0].replace('\\n', '').replace('\n', '').replace('\\', '')
        context_list.append(context) ## add context
        
        target = text.split('\n\nColumn 2:')[1].split('\n\nTake these examples as reference:')[0].replace('\\n', '').replace('\n', '').replace('\\', '')
        target_list.append(target)
    return context_list,target_list,label_list

def AVE_Embed_Text(df): ## 返回3列分别是text_1,text_2,label
    SimTab = df
    context_list = []
    target_list = []
    label_list = []
    for i in tqdm(range(len(SimTab))):
        text = SimTab.iloc[i,0]
        label = SimTab.iloc[i,2]
        label_list.append(label) ## add label
        context = text.split('product title:\n\n')[1].split('\n\nTake these rows as examples:')[0]
        context_list.append(context) ## add context
        
        # context_dict = eval(context)
        
        target = text.split('Output Format Example:\n\n')[1].split('\n\nproduct title:')[0]
        
        # target_attr = list(eval(target).keys())[0]
        
        # related_context = context_dict[target_attr]
        target_list.append(target)

    return context_list,target_list,label_list
args = load_yaml_args(yaml_path)
# print(args.train_init_model)


cluster_num = args.cluster_num
sample_per_cluster = args.sample_per_cluster
train_file_path = args.train_file_path
device = args.devices
task = args.task
embedding_model_path = args.embedding_model_path
# ppl_path = args.ppl_path
batch_size = args.batch_size
if isinstance(device,str):
    device_list = device.split(',')
else:
    device_list = [str(device)]
print(f'Run on CUDA:{device}')
time_dict = {}

train_file = pd.read_json(train_file_path)

start_time = time.time()


if(task=='CTA'):
    left_list,right_list,label_list = CTA_Embed_Text(train_file) ## 返回3个list
elif(task=='RE'):
    left_list,right_list,label_list = RE_Embed_Text(train_file) ## 返回3个list
elif(task=='ER'):
    left_list,right_list,label_list = ER_Embed_Text(train_file) ## 返回3个list
elif(task=='DC'):
    left_list,right_list,label_list = DC_Embed_Text(train_file) ## 返回3个list
elif(task=='DI'):
    left_list,right_list,label_list = DI_Embed_Text(train_file) ## 返回3个list
elif(task=='SM'):
    left_list,right_list,label_list = SM_Embed_Text(train_file) ## 返回3个list
elif(task=='AVE'):
    left_list,right_list,label_list = AVE_Embed_Text(train_file) ## 返回3个list
## Set only One-GPU for embedding model

print(len(np.unique(label_list)))

os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(device_list[-1]) ## only use last device for embedding model

model = FlagModel(embedding_model_path, 
                  use_fp16=True)

embedding_a = model.encode(left_list)
embedding_b = model.encode(right_list)
embedding_c = model.encode(label_list)

## Calculate PPL, need modify for random calculation and .csv conditions

if args.require_ppl:
    ## 如果 ppl-path 非空
    if not os.path.exists('ppl/{}/{}/ppl-init-{}.csv'.format(args.task,args.dataset,1)): 
        command_IF = []
        for process_num in range(len(device_list)): ## 从0开始
            command_IF.append('CUDA_VISIBLE_DEVICES={} python cal_ppl_mp.py --yaml_path {} --process_num {} --total_process_num {}'.format(device_list[process_num], yaml_path, process_num+1,len(device_list)))

        # if not hasattr(mp, '_start_method'):
        #     mp.set_start_method('spawn')
        processes = []
        for command in command_IF:
            p = mp.Process(target=run_command, args=(command,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print("所有命令执行完毕，进行下一步")
    
    ppl_array = np.zeros(len(train_file))
    for process_num in range(1,9,1): ## maximum of k process
        if os.path.exists('ppl/{}/{}/ppl-init-{}.csv'.format(args.task,args.dataset,process_num)): ## i-th gradient 
            ppl_df = pd.read_csv('ppl/{}/{}/ppl-init-{}.csv'.format(args.task,args.dataset,process_num),index_col=0)
            for index,row in ppl_df.iterrows():
                ppl_array[index] = row[0]
else:
    ppl = pd.read_json(args.ppl_path)
    ppl_list = ppl.iloc[:,0].to_list()
    ppl_array = np.array(ppl_list)




pt_data = np.concatenate([embedding_a,embedding_b,embedding_c],axis=1)

## Calculate Clustering for init

high_dim_vectors = pt_data


## Clustering high_dim_vectors
clustering = do_clustering(high_dim_vectors,kmeans_num_clusters=cluster_num)
cluster_labels = clustering.labels_

## Initial Data Selection

middle_confidence_samples = sample_middle_confidence_data(cluster_labels, ppl_array, n = sample_per_cluster, low_th=25, up_th = 75)

new_data = get_json_sample(middle_confidence_samples)

init_df = train_file.iloc[new_data]

if(task=='SM'):
    train_file_pos = train_file[train_file['output']=="{'Output': 'match'}"]
    init_df = pd.concat([init_df,train_file_pos])
    print('add positive samples')


if args.train_init_model:
    create_folder_for_file('train/{}/{}/train-init.json'.format(args.task,args.dataset))
    json.dump(init_df.to_dict(orient='records'), open('train/{}/{}/train-init.json'.format(args.task,args.dataset), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    torch.save(middle_confidence_samples,'train/{}/{}/train-init.pkl'.format(args.task,args.dataset))
    
end_time = time.time()
time_dict['init_select'] = end_time - start_time
np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy.npy',time_dict)
### train init model

yaml_template_path = 'script/qwen_lora_CTA_SimTab-P1.yaml'

with open(yaml_template_path, 'r') as file:
    data = yaml.safe_load(file)

# 修改特定属性，这里以修改某个键的值为例，你可以根据实际需求调整修改逻辑
data['cutoff_len'] = args.cutoff_len
data['train_file_path'] = train_file_path
data['output_dir'] = 'lora/qwen-0.5B/{}/{}/init'.format(args.task,args.dataset)
lora_dir = 'lora/qwen-0.5B/{}/{}/init'.format(args.task,args.dataset)
# 打开新的yaml文件，将修改后的数据写入
with open('script/qwen_lora_{}_{}_init.yaml'.format(args.task,args.dataset), 'w') as file:
    yaml.dump(data, file)

## Run Command
if not os.path.exists(lora_dir):
    run_command('CUDA_VISIBLE_DEVICES={} llamafactory-cli train script/qwen_lora_{}_{}_init.yaml'.format(device,args.task,args.dataset))

end_time = time.time()
time_dict['init-train'] = end_time - start_time
np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy.npy',time_dict)
## Calculate FL Score

cluster_indices = {i: np.where(cluster_labels == i)[0] for i in range(cluster_num)}

## Re-Organize Batch via FL score, for batch-IF Calculation

batch_division = {}
FL_Score = {}
greedyList_All = {}
batch_sampler = []

for i in range(cluster_num):
    batch_division[i] = []
    cluster_indice_index = cluster_indices[i]
    high_dim_vectors_cluster = high_dim_vectors[cluster_indice_index]
    cluster_size = len(cluster_indice_index)
    # print(i,cluster_size)
    fla_num = int(np.ceil(cluster_size / batch_size))

    
    try:
        idx_list,greedyList = do_fla(high_dim_vectors_cluster,high_dim_vectors_cluster.shape[0],number_select=fla_num) ## idx_list is the selected number
        result,coverage = cluster_vectors(high_dim_vectors_cluster,idx_list,batch_size)
        for cluster_ind in result.keys():
            global_result = [cluster_indice_index[j] for j in result[cluster_ind]] ## 将相对index映射到global index
            batch_division[i].append(global_result)
    except: ## AgglomerativeClustering not support single element
        batch_division[i].append(cluster_indice_index)
        print('{}-th cluster element index: {}'.format(i,cluster_indice_index))
        
    ### 全局FL分数
    _,greedyList_AllElement = do_fla(high_dim_vectors_cluster,high_dim_vectors_cluster.shape[0],number_select=high_dim_vectors_cluster.shape[0]-1)
    greedyList_AllElement_dict = {}
    for (index,value) in greedyList_AllElement: ## 遍历fla选取
        global_index = cluster_indice_index[index] ## 从相对index转为全局index
        greedyList_AllElement_dict[global_index] = value 
    greedyList_All[i] = greedyList_AllElement_dict
    
for key in batch_division.keys():
    batch_sampler.extend(batch_division[key])

## Save Batch Division Result
create_folder_for_file('Influence/{}/{}/batch-size-{}/batch.pkl'.format(args.task,args.dataset,batch_size))
torch.save(batch_sampler,'Influence/{}/{}/batch-size-{}/batch.pkl'.format(args.task,args.dataset,batch_size))

end_time = time.time()
time_dict['batch-division'] = end_time - start_time
np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy',time_dict)

remove_dir_if_exists(f'grad/{args.task}/{args.dataset}')

## 根据显卡数量分配线程，可能可以修改
## TODO:config.yaml没有在--args内，需要修改
if args.require_grad:
    command_IF = []
    for process_num in range(len(device_list)): ## 从0开始
        command_IF.append('CUDA_VISIBLE_DEVICES={} python cal_IF_mp.py --yaml_path {} --process_num {} --total_process_num {}'.format(device_list[process_num],yaml_path,process_num+1,len(device_list)))

    # if not hasattr(mp, '_start_method'):
    #     mp.set_start_method('spawn')
    processes = []
    for command in command_IF:
        p = mp.Process(target=run_command, args=(command,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("所有命令执行完毕，进行下一步")

end_time = time.time()
time_dict['gradient-calculation'] = end_time - start_time
np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy',time_dict)
# command_ppl = 'CUDA_VISIBLE_DEVICES={} python cal_IF_self_divide.py  \
#     --yaml_path {}\
#     --process_num {}\
#     --total_process_num {}'.format(device,'config.yaml',1,1)

print('python IF_Cal_mp.py  --device cuda --task {} --dataset {} --batch_size {} --save_all_layers'.format(args.task,args.dataset,batch_size))

run_command('python IF_Cal_mp.py  --device cuda --task {} --dataset {} --batch_size {} --save_all_layers'.format(args.task,args.dataset,batch_size))

end_time = time.time()
time_dict['IF-Score'] = end_time - start_time
np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy.npy',time_dict)
## IF Score
sample_IF = torch.load('Influence/{}/{}/batch-size-{}/score.pkl'.format(args.task,args.dataset,batch_size),weights_only=False)
sample_IF = z_score_normalize(sample_IF)



## FL Score
greedyList_All_norm = z_score_normalize(greedyList_All)
## Flatten
greedyList_All_norm_flatten = {}
for key in greedyList_All_norm.keys():
    for index in greedyList_All_norm[key]:
        greedyList_All_norm_flatten[index] = greedyList_All_norm[key][index]

total_score = np.zeros(len(train_file))
## calculate ppl
# for index,row in ppl.iterrows():
#     total_score[index] += row[0]

for index in range(len(total_score)):
    total_score[index] = ppl_array[index]
## add FL
for key in greedyList_All_norm_flatten.keys():
    total_score[key] += greedyList_All_norm_flatten[key]
## add global_IF
for key in sample_IF['iterative']:
    total_score[key] -= sample_IF['iterative'][key] ## Influence Score与performance成反比，所以是-=
    
## 遍历cluster,排序
cluster_rank = {}
for i in range(cluster_num): ## 100 is the cluster number, hyper-parameter
    cluster_rank[i] = {}
    cluster_index = cluster_indices[i]
    sorted_index = cluster_index[np.argsort(-total_score[cluster_index])] ## return to global index
    cluster_rank[i] = sorted_index

p2_choose_index = []
cluster_per_budget = sample_per_cluster
for i in range(cluster_num):
    p2_choose_index.extend(cluster_rank[i][:cluster_per_budget])

## save FL Score for ablation
create_folder_for_file('selection/{}/{}/batch-size-{}/FL-Score.pkl'.format(args.task,args.dataset,batch_size))
torch.save(greedyList_All_norm_flatten,'selection/{}/{}/batch-size-{}/FL-Score.pkl'.format(args.task,args.dataset,batch_size))
## save total score for ablation
torch.save(total_score,'selection/{}/{}/batch-size-{}/Total-Score.pkl'.format(args.task,args.dataset,batch_size))
## save cluster division for ablation and furthur experiment
torch.save(cluster_rank,'selection/{}/{}/batch-size-{}/Cluster-Rank.pkl'.format(args.task,args.dataset,batch_size))

## output

selected_df = train_file.iloc[p2_choose_index]

if(task=='SM'):
    train_file_pos = train_file[train_file['output']=="{'Output': 'match'}"]
    selected_df = pd.concat([selected_df,train_file_pos])
    print('add positive samples')

json.dump(selected_df.to_dict(orient='records'), open('train/{}/{}/train-select-w-main-batch-size-{}.json'.format(args.task,args.dataset,batch_size), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

end_time = time.time()
time_dict['Final Selection'] = end_time - start_time
np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy',time_dict)
### 使用selected data完成training

# train_yaml_template_path = 'script/mistral_lora_RE-RE-P2.yaml'

# with open(train_yaml_template_path, 'r') as file:
#     train_args = yaml.safe_load(file)

# train_args['train_file_path'] = 'train/{}/{}/train-select.json'.format(args.task,args.dataset) ## 指定训练数据位置
# train_args['output_dir'] = 'lora/mistral-7B/{}/{}/select'.format(args.task,args.dataset) ## 指定lora输出位置

# with open('script/mistral_lora_{}_{}_P2.yaml'.format(args.task,args.dataset), 'w') as file:
#     yaml.dump(train_args, file)

# if not os.path.exists(train_args['output_dir']):
#     run_command('CUDA_VISIBLE_DEVICES={} llamafactory-cli train script/mistral_lora_{}_{}_P2.yaml'.format(device,args.task,args.dataset))
    
# end_time = time.time()
# time_dict['Selection fine-tune'] = end_time - start_time
# np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy.npy',time_dict)
# ### 查询

# def round_down_to_power_of_two(num): ## vllm查询仅支持1/2/4/8的tensor_parallel
#     """
#     将给定的整数向下取整到最接近的1、2、4、8中的某个值。
#     """
#     result = 1
#     while result < num:
#         result *= 2
#         if result > num:
#             result //= 2
#     return result

# if args.require_inference:
#     if args.guided_choices:
#         command = 'CUDA_VISIBLE_DEVICES={} python vllm_inference_mistral_api.py --model_path {} --directory {} --gpu_num {} --file {} --json --guided_choices'.format(
#             device,
#             train_args['model_name_or_path'],
#             train_args['output_dir'],
#             round_down_to_power_of_two(len(device_list)),
#             args.test_file_path
#         )
#     else:
#         command = 'CUDA_VISIBLE_DEVICES={} python vllm_inference_mistral_api.py --model_path {} --directory {} --gpu_num {} --file {} --json'.format(
#             device,
#             train_args['model_name_or_path'],
#             train_args['output_dir'],
#             round_down_to_power_of_two(len(device_list)),
#             args.test_file_path
#         )
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)

#     print("标准输出：", result.stdout)
# end_time = time.time()
# time_dict['Selection Inference'] = end_time - start_time

# print(time_dict)
# # np.save(f'eval_result/time_dict_{yaml_path}.npy',time_dict)

# np.save(f'eval_result/time_dict_{args.task}_{args.dataset}_{device}_batch-size-{batch_size}.npy.npy',time_dict)


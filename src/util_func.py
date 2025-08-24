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
# import submodlib
# from submodlib.functions.facilityLocation import FacilityLocationFunction
from datasets import Dataset
import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import yaml
import argparse

import torch.multiprocessing as mp
import subprocess

yaml_path = 'script/config_CTA_SimTab.yaml'

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

def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        args_dict = yaml.safe_load(file)
    return args_dict

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


def get_top_k_indices(data_dict: dict, k: int, IF = False) -> list:
    """
    从字典中获取前 K 个最大数值对应的索引。

    Args:
        data_dict (dict): 输入字典，key 为索引，value 为数值。
                          例如：{0: 10, 1: 5, 2: 20, 3: 8}
        k (int): 要获取的最大元素的数量。

    Returns:
        list: 包含前 K 个最大数值对应索引的列表，按数值从大到小排序。
              如果 k 大于字典的元素数量，将返回所有索引。
    """
    if not isinstance(data_dict, dict):
        raise TypeError("Input 'data_dict' must be a dictionary.")
    if not isinstance(k, int) or k < 0:
        raise ValueError("Input 'k' must be a non-negative integer.")

    if k == 0:
        return []
    if not data_dict:
        return []

    # 使用 heapq.nlargest 获取前 K 个最大的 (value, key) 对
    # nlargest 默认按第一个元素（value）排序
    # 注意：nlargest 返回的顺序是数值从大到小
    if not IF: ## 如果按照降序排列，默认为True
        top_k_items = heapq.nlargest(k, data_dict.items(), key=lambda item: item[1])
    elif IF: ## 如果按照升序排列
        top_k_items = heapq.nsmallest(k, data_dict.items(), key=lambda item: item[1])        

    # 提取索引（key）并返回
    # item[0] 是原始的 key
    top_k_indices = [item[0] for item in top_k_items]

    return top_k_indices
def round_down_to_power_of_two(num: int): 
    if num == 1:
        return 1
    elif num >=2 and num < 4:
        return 2
    elif num >=4 and num < 8:
        return 4
    else:
        return 8
def normalize_to_neg1_pos1(arr: np.ndarray) -> np.ndarray:
    """
    将 numpy 数组归一化到 [-1, 1] 区间。

    参数:
        arr (np.ndarray): 输入数组

    返回:
        np.ndarray: 归一化后的数组，范围在 [-1, 1]
    """
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        # 避免除以0，返回全零数组
        return np.zeros_like(arr)
    else:
        return 2 * (arr - min_val) / (max_val - min_val) - 1
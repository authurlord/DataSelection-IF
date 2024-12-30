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


args = load_yaml_args('config.yaml')
print(args.train_init_model)


cluster_num = args.cluster_num
sample_per_cluster = args.sample_per_cluster
train_file_path = args.train_file_path
device = args.devices
embedding_model_path = args.embedding_model_path
ppl_path = args.ppl_path
batch_size = args.batch_size
device_list = device.split(',')

train_file = pd.read_json(train_file_path)



left_list,right_list,label_list = CTA_Embed_Text(train_file) ## 返回3个list

## Set only One-GPU for embedding model

os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(device_list[-1]) ## only use last device for embedding model

model = FlagModel(embedding_model_path, 
                  use_fp16=True)

embedding_a = model.encode(left_list)
embedding_b = model.encode(right_list)
embedding_c = model.encode(label_list)

## Calculate PPL, need modify for random calculation and .csv conditions
ppl = pd.read_json(ppl_path)
ppl_list = ppl.iloc[:,0].to_list()


pt_data = np.concatenate([embedding_a,embedding_b,embedding_c],axis=1)

## Calculate Clustering for init

high_dim_vectors = pt_data
ppl_array = np.array(ppl_list)

## Clustering high_dim_vectors
clustering = do_clustering(high_dim_vectors,kmeans_num_clusters=cluster_num)
cluster_labels = clustering.labels_

## Initial Data Selection

middle_confidence_samples = sample_middle_confidence_data(cluster_labels, ppl_array, n = sample_per_cluster, low_th=25, up_th = 75)

new_data = get_json_sample(middle_confidence_samples)

init_df = train_file.iloc[new_data]

if args.train_init_model:
    create_folder_for_file('train/{}/{}/train-init.json'.format(args.task,args.dataset))
    json.dump(init_df.to_dict(orient='records'), open('train/{}/{}/train-init.json'.format(args.task,args.dataset), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    torch.save(middle_confidence_samples,'train/{}/{}/train-init.pkl'.format(args.task,args.dataset))


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
    idx_list,greedyList = do_fla(high_dim_vectors_cluster,high_dim_vectors_cluster.shape[0],number_select=fla_num) ## idx_list is the selected number
    
    try:
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
create_folder_for_file('Influence/{}/{}/batch.pkl'.format(args.task,args.dataset))
torch.save(batch_sampler,'Influence/{}/{}/batch.pkl'.format(args.task,args.dataset))

## 根据显卡数量分配线程，可能可以修改
## TODO:config.yaml没有在--args内，需要修改
command_IF = []
for process_num in range(len(device_list)): ## 从0开始
    command_IF.append('CUDA_VISIBLE_DEVICES={} python cal_IF_mp.py --yaml_path config.yaml --process_num {} --total_process_num {}'.format(device_list[process_num],process_num+1,len(device_list)))

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

# command_ppl = 'CUDA_VISIBLE_DEVICES={} python cal_IF_self_divide.py  \
#     --yaml_path {}\
#     --process_num {}\
#     --total_process_num {}'.format(device,'config.yaml',1,1)


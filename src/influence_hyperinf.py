from time import time
from tqdm.notebook import tqdm
import sys
from collections import defaultdict
import pandas as pd
import pickle, os
import torch
import numpy as np
### Influence Function Computation by different methods
class IFEngine(object):
        # 初始化存储时间、Hessian-Vector Product (HVP) 结果和影响函数（Influence Function）值的字典

    def __init__(self):
        self.time_dict=defaultdict(list)
        self.hvp_dict=defaultdict(list)
        self.IF_dict=defaultdict(list)

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict, noise_index=None): # 存储训练集和验证集的梯度字典，并计算验证集平均梯度
        self.tr_grad_dict = tr_grad_dict # 训练集梯度字典
        self.val_grad_dict = val_grad_dict # 验证集梯度字典
        self.noise_index = noise_index # 可选的噪声索引（用于异常值处理）
        # 存储训练集和验证集的样本数
        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())
        self.compute_val_grad_avg() # 计算验证集平均梯度

    def compute_val_grad_avg(self):
        # Compute the avg gradient on the validation dataset
        self.val_grad_avg_dict={}
        for weight_name in self.val_grad_dict[0]:
            # 初始化为与梯度同设备的零向量
            self.val_grad_avg_dict[weight_name]=torch.zeros(self.val_grad_dict[0][weight_name].shape).to(self.val_grad_dict[0][weight_name].device)
            # 逐个样本累加梯度并取平均值
            for val_id in self.val_grad_dict:
                self.val_grad_avg_dict[weight_name] += self.val_grad_dict[val_id][weight_name] / self.n_val

    def compute_hvps(self, lambda_const_param=10, compute_accurate=True, compute_LiSSA=True):
        '''
        Compute the influence function score under each method
        '''
        self.compute_hvp_iterative(lambda_const_param=lambda_const_param) ## HyperInf,使用 Schulz 迭代法计算 HVP 和近似 Hessian 逆矩阵
        self.compute_hvp_identity() ## Baseline TracIn
        self.compute_hvp_proposed(lambda_const_param=lambda_const_param) ## Datainf
        if compute_LiSSA:
            self.compute_hvp_LiSSA(lambda_const_param=lambda_const_param)
        if compute_accurate:
            self.compute_hvp_accurate(lambda_const_param=lambda_const_param)

    def compute_hvp_identity(self):
        '''
        TracIN
        '''
        start_time = time()
        self.hvp_dict['identity'] = self.val_grad_avg_dict.copy()
        self.time_dict['identity'] = time()-start_time
        print("Time taken for Hessian-free: ", self.time_dict['identity'])

    def compute_hvp_iterative(self, lambda_const_param=10, n_iteration=30):
        '''
        Compute the influence funcion score by our method HyperINF
        '''

        def schulz_inverse_stable(A, damping_factor=0, max_iterations=20, tol=1e-6):
            n = A.shape[0]
            #I = np.eye(n)
            I = torch.eye(n, device=A.device)
            A_damped = A + damping_factor * I  # Apply damping for stable

            #X = np.eye(n) * 0.00005  # Initial estimate of inverse matrix
            X = torch.eye(n, device=A.device) * 0.00005  # Initial estimate of inverse matrix

            for _ in range(max_iterations):
                # 通过 Schulz 方法更新逆矩阵估计值
                #X = X.dot(2 * I - A_damped.dot(X))
                X = X @ (2 * I - A_damped @ X)

                # # Check for convergence
                # if np.linalg.norm(I - A.dot(X)) < tol:
                #     break

            return X

        start_time = time()
        hvp_iterative_dict={}

        for _, weight_name in enumerate(tqdm(self.val_grad_avg_dict)):
            # lambda_const computation = 0.1 x (n * d_l)^(-1) \sum_{i=1}^{n} ||grad_i^l||_2^2, 计算 λ（正则化系数）：基于梯度的方差
            S=torch.zeros(len(self.tr_grad_dict.keys())).to(self.val_grad_avg_dict[weight_name].device)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # iterative hvp computation
            # G_l: same shape of self.tr_grad_dict[0][weight_name].T @ self.tr_grad_dict[0][weight_name]
            # 构造 Fisher 信息矩阵（近似 Hessian）
            G_l = torch.zeros((self.tr_grad_dict[0][weight_name].T @ self.tr_grad_dict[0][weight_name]).shape).to(self.val_grad_avg_dict[weight_name].device)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device) # (grad_i^l)^T
                G_l += tmp_grad.T @ tmp_grad / self.n_train
                
            # 加入正则化项以保证矩阵可逆
            G_l = G_l + lambda_const * torch.eye(G_l.shape[0], device=G_l.device)
           # G_l = G_l.cpu().detach().numpy()
           # 使用 Schulz 方法近似计算逆矩阵
            G_l_inv = schulz_inverse_stable(G_l, damping_factor=0.001, max_iterations=n_iteration, tol=1e-6)
            # 计算 HVP: G_l_inv @ val_grad_avg
            hvp_iterative_dict[weight_name] = torch.tensor(self.val_grad_avg_dict[weight_name] @ G_l_inv)
            #print(hvp_iterative_dict[weight_name])
        self.hvp_dict['iterative'] = hvp_iterative_dict
        self.time_dict['iterative'] = time()-start_time
        print("Time taken for HyperINF: ", self.time_dict['iterative'])



    def compute_hvp_proposed(self, lambda_const_param=10):
        '''
        DataInf method
        '''
        start_time = time()
        hvp_proposed_dict={}

        for _ , weight_name in enumerate(tqdm(self.val_grad_avg_dict)):
            # lambda_const computation = 0.1 x (n * d_l)^(-1) \sum_{i=1}^{n} ||grad_i^l||_2^2
            S=torch.zeros(len(self.tr_grad_dict.keys())).to(self.val_grad_avg_dict[weight_name].device)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device)
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation
            hvp=torch.zeros(self.val_grad_avg_dict[weight_name].shape).to(self.val_grad_avg_dict[weight_name].device)
            for tr_id in self.tr_grad_dict: # i
                tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device) # grad_i^l
                # L_(l,i) / (lambda + ||grad_i^l||_2^2) in Eqn. (5)
                C_tmp = torch.sum(self.val_grad_avg_dict[weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2)).to(self.val_grad_avg_dict[weight_name].device)
                # (v_l^T - C_tmp * (grad_i^l)^T ) / (n * lambda) in Eqn. (5)
                hvp += (self.val_grad_avg_dict[weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
            hvp_proposed_dict[weight_name] = hvp
        self.hvp_dict['proposed'] = hvp_proposed_dict
        self.time_dict['proposed'] = time()-start_time
        print("Time taken for Datainf: ", self.time_dict['proposed'])

    def compute_hvp_accurate(self, lambda_const_param=10):
        start_time = time()
        hvp_accurate_dict={}
        for _ , weight_name in enumerate(tqdm(self.val_grad_avg_dict)):

            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation (eigenvalue decomposition)
            AAt_matrix = torch.zeros(torch.outer(self.tr_grad_dict[0][weight_name].reshape(-1),
                                                 self.tr_grad_dict[0][weight_name].reshape(-1)).shape)
            for tr_id in self.tr_grad_dict:

                tmp_mat = torch.outer(self.tr_grad_dict[tr_id][weight_name].reshape(-1),
                                      self.tr_grad_dict[tr_id][weight_name].reshape(-1))
                AAt_matrix += tmp_mat


            L, V = torch.linalg.eig(AAt_matrix)
            L, V = L.float(), V.float()
            hvp = self.val_grad_avg_dict[weight_name].reshape(-1) @ V
            hvp = (hvp / (lambda_const + L/ self.n_train)) @ V.T

            hvp_accurate_dict[weight_name] = hvp.reshape(len(self.tr_grad_dict[0][weight_name]), -1)
            del tmp_mat, AAt_matrix, V # to save memory
        self.hvp_dict['accurate'] = hvp_accurate_dict
        self.time_dict['accurate'] = time()-start_time
        print("Time taken for Accurate: ", self.time_dict['accurate'])

    def compute_hvp_LiSSA(self, lambda_const_param=10, n_iteration=10, alpha_const=1.):
        '''
        LiSSA method
        '''
        start_time = time()
        hvp_LiSSA_dict={}

        for _, weight_name in enumerate(tqdm(self.val_grad_avg_dict)):
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys())).to(self.val_grad_avg_dict[weight_name].device)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device)
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation
            running_hvp=self.val_grad_avg_dict[weight_name]
            for _ in range(n_iteration):
                hvp_tmp=torch.zeros(self.val_grad_avg_dict[weight_name].shape).to(self.val_grad_avg_dict[weight_name].device)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device)
                    hvp_tmp += (torch.sum(tmp_grad*running_hvp)*tmp_grad - lambda_const*running_hvp) / self.n_train

                running_hvp = self.val_grad_avg_dict[weight_name] + running_hvp - alpha_const*hvp_tmp
            hvp_LiSSA_dict[weight_name] = running_hvp

        self.hvp_dict['LiSSA'] = hvp_LiSSA_dict
        self.time_dict['LiSSA'] = time()-start_time
        print("Time taken for LiSSA: ", self.time_dict['LiSSA'])

    def compute_IF(self):
        for method_name in self.hvp_dict:
            if_tmp_dict = {}
            for tr_id in self.tr_grad_dict:
                if_tmp_value = 0
                for weight_name in self.val_grad_avg_dict:
                    if_tmp_value += torch.sum(self.hvp_dict[method_name][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                if_tmp_dict[tr_id]= -if_tmp_value.cpu()
               # print(-if_tmp_value)

            self.IF_dict[method_name] = pd.Series(if_tmp_dict, dtype=float).to_numpy()

    def save_result(self, noise_index, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['noise_index']=noise_index
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)

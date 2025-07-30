import pandas as pd
import numpy as np
from collections import defaultdict

class ReTaskEvaluator:
    """
    一个用于评估特定任务（如关系抽取 'RE'）结果的类。

    该类接收一个包含模型预测结果的 DataFrame，并根据指定的任务和数据集名称，
    执行相应的评估逻辑，最终计算出准确率（acc）、微平均F1（micro_f1）和
    宏平均F1（macro_f1）等指标。
    """

    def __init__(self, df: pd.DataFrame, task: str, dataset: str):
        """
        初始化评估器。

        Args:
            df (pd.DataFrame): 包含模型预测结果的 DataFrame。
                               对应原始脚本中的 `SimTab_test`。
                               必须包含 'output' 和 'predict' 列。
            task (str): 任务名称，例如 'RE'。
            dataset (str): 数据集名称，例如 'RE'。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 参数必须是一个 pandas DataFrame。")
        self.df = df
        self.task = task
        self.dataset = dataset

    def _ast_transform(self, row: pd.Series) -> tuple:
        """
        一个私有辅助方法，用于从 DataFrame 的行中解析出真实值和预测值。
        对应原始脚本中的 `Ast` 函数。
        """
        # 解析真实值，假设它总是在'output'列的第一个值
        try:
            truth = list(eval(row['output']).values())[0]
        except (SyntaxError, IndexError, TypeError):
            truth = [] # 如果解析失败，返回空列表

        # 解析预测值，使用 try-except 以处理可能的解析错误
        try:
            pred = list(eval(row['predict']).values())[0]
        except (SyntaxError, IndexError, TypeError, NameError):
            pred = '' # 如果解析失败，返回空字符串
        
        index = row['index']
        return truth, pred, index

    def _calculate_f1_scores(self, pred_list: list, truth_list: list) -> dict:
        """
        根据给定的预测列表和真实值列表计算微平均F1和宏平均F1。
        这是一个私有辅助方法。
        """
        # 获取所有唯一的类别/标签
        all_labels = set(p for p in pred_list) | set(t for tl in truth_list for t in tl)

        # 初始化 TP, FP, FN 计数器
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        # 遍历每个预测和真实值对
        for pred, truths in zip(pred_list, truth_list):
            if pred in truths:
                tp[pred] += 1
            else:
                fp[pred] += 1
            for true_label in truths:
                if pred != true_label:
                    fn[true_label] += 1

        # 计算 Micro-F1
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        # 计算 Macro-F1
        f1_scores_per_class = []
        for label in all_labels:
            label_tp = tp[label]
            label_fp = fp[label]
            label_fn = fn[label]
            precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) > 0 else 0
            recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores_per_class.append(f1)
        macro_f1 = sum(f1_scores_per_class) / len(f1_scores_per_class) if f1_scores_per_class else 0

        return {'micro_f1': micro_f1, 'macro_f1': macro_f1}

    def evaluate(self, 
                 all_relation_path: str = 'train/RE/all_relation.npy',
                 test_rag_path: str = 'train/RE/test_RAG.csv') -> dict:
        """
        执行评估流程。如果任务和数据集名称匹配 'RE'，则运行评估，否则跳过。

        Args:
            all_relation_path (str, optional): `all_relation.npy` 文件的路径。
            test_rag_path (str, optional): `test_RAG.csv` 文件的路径。

        Returns:
            dict: 一个包含评估指标的字典 (acc, micro_f1, macro_f1)。
                  如果评估被跳过或发生错误，则返回 None。
        """
        if self.task != 'RE' or self.dataset != 'RE':
            print(f"评估跳过：任务 '{self.task}' 或数据集 '{self.dataset}' 不是 'RE'。")
            return None

        try:
            # 加载评估所需的外部文件
            all_relation = np.load(all_relation_path, allow_pickle=True)
            re_test = pd.read_csv(test_rag_path, index_col=0)
        except FileNotFoundError as e:
            print(f"错误：无法加载评估文件。请检查路径。 {e}")
            return None

        # --- 数据转换 ---
        # 使用 self.df 的副本进行操作，以避免修改原始 DataFrame
        sim_tab_test = self.df.copy()
        sim_tab_test['index'] = sim_tab_test.index
        
        # 应用转换函数
        sim_tab_test_transform = sim_tab_test.apply(self._ast_transform, axis=1, result_type='expand')
        sim_tab_test_transform.columns = ['truth', 'pred', 'index']

        # --- 评估循环 ---
        truth_list = []
        pred_list = []
        unique_indices = sim_tab_test_transform['index'].unique()

        for i in unique_indices:
            select_df = sim_tab_test_transform[sim_tab_test_transform['index'] == i]
            
            # 从外部文件中获取真实的标签
            try:
                truth = eval(re_test.iloc[i, 1])
            except (IndexError, SyntaxError):
                print(f"警告：无法在索引 {i} 处从 {test_rag_path} 获取真实值。跳过此项。")
                continue

            # 筛选出在 `all_relation` 中的预测
            select_df_filter = select_df[select_df['pred'].isin(all_relation)]
            
            # 确定最终预测值（多数投票）
            try:
                # 优先从筛选后的预测中选择
                pred = select_df_filter['pred'].value_counts().idxmax()
            except ValueError:
                # 如果筛选后为空，则从所有预测中选择
                try:
                    pred = select_df['pred'].value_counts().idxmax()
                except ValueError:
                    pred = "" # 如果没有任何预测，则为空字符串
            
            truth_list.append(truth)
            pred_list.append(pred)

        # --- 指标计算 ---
        if not truth_list:
            print("警告：没有可供评估的数据。")
            return {'acc': 0, 'micro_f1': 0, 'macro_f1': 0}

        count = sum(1 for pred, truth in zip(pred_list, truth_list) if pred in truth)
        
        result_metric = {}
        result_metric['acc'] = count / len(truth_list)

        f1_metric = self._calculate_f1_scores(pred_list=pred_list, truth_list=truth_list)
        result_metric.update(f1_metric)

        return result_metric
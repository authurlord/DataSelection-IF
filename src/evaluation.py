from sklearn.metrics import precision_score,recall_score,f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from collections import defaultdict

def calculate_f1_scores(pred_list, truth_list):
    """
    根据给定的预测列表和真实值列表计算微平均F1 (micro-F1) 和宏平均F1 (macro-F1)。

    这个函数适用于每个预测对应一个真实值列表（其中包含一个或多个正确答案）的场景。

    Args:
        pred_list (list): 包含 n 个预测元素的列表。
        truth_list (list): 包含 n 个列表的列表，每个内层列表包含一个或多个正确的真实值。

    Returns:
        dict: 一个字典，包含 'micro_f1' 和 'macro_f1' 两个键值对。
              {'micro_f1': float, 'macro_f1': float}
    """
    # 步骤 1: 获取所有唯一的类别/标签
    all_labels = set(p for p in pred_list) | set(t for tl in truth_list for t in tl)

    # 步骤 2: 初始化 TP, FP, FN 计数器
    # 使用 defaultdict 可以让代码更简洁，当一个键第一次被访问时，会自动创建默认值（这里是0）。
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # 步骤 3: 遍历每个预测和真实值对，填充 TP, FP, FN 计数器
    for pred, truths in zip(pred_list, truth_list):
        # 检查预测是否正确，来更新 TP 和 FP
        # pred: 当前的预测值
        # truths: 当前预测值对应的真实值列表
        if pred in truths:
            # 预测正确，该预测类别的 TP 加 1
            tp[pred] += 1
        else:
            # 预测错误，该预测类别的 FP 加 1
            fp[pred] += 1

        # 检查是否有漏掉的正确答案，来更新 FN
        for true_label in truths:
            # 如果真实值列表中的某个标签不等于我们的预测值，
            # 那么对于那个标签来说，这是一个 False Negative。
            if pred != true_label:
                fn[true_label] += 1

    # 步骤 4: 计算 Micro-F1 (微平均)
    # 首先，全局汇总所有类别的 TP, FP, FN
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    # 计算微平均 Precision 和 Recall
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    # 计算微平均 F1
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # 步骤 5: 计算 Macro-F1 (宏平均)
    f1_scores_per_class = []
    # 遍历所有出现过的类别
    for label in all_labels:
        # 获取该类别的 TP, FP, FN
        label_tp = tp[label]
        label_fp = fp[label]
        label_fn = fn[label]

        # 计算该类别的 Precision 和 Recall
        precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) > 0 else 0
        recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) > 0 else 0

        # 计算该类别的 F1 分数
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores_per_class.append(f1)

    # 宏平均 F1 是所有类别 F1 分数的算术平均值
    macro_f1 = sum(f1_scores_per_class) / len(f1_scores_per_class) if f1_scores_per_class else 0

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }
def Transfer(row):
    if(row['output'].__contains__('mismatch')) or (row['output'].__contains__('dismatch')):
        label = 0
    else:
        label = 1
    if(row['prediction'].__contains__('mismatch')) or (row['prediction'].__contains__('dismatch')):
        predict = 0
    else:
        predict = 1
    return label,predict
def AVE_AST(row):
    output = row['output'].strip()
    predict = row['predict'].strip()
    output_item = list(ast.literal_eval(output).values())[0]
    predict_item = list(ast.literal_eval(predict).values())[0]
    row['output_item'] = output_item.lower()
    row['predict_item'] = predict_item.lower()
    return row
def Ast(row):
    truth = list(eval(row['output']).values())[0]
    try:
        pred = list(eval(row['predict']).values())[0]
    except:
        pred = ''
    index = row['index']
    return truth, pred, index
def Str2Int(row):
    for index in range(11):
        temp = row[index]
        try:
            row[index] = str(int(temp))
        except:
            continue
    return row
def try_convert_to_int(row):
    for x,y in row.items():
        if(x in ['ounces','ibu']):
            try:
                row[x] = int(y)
            except:
                row[x] = y
    return row
def Impuration_AST(row):
    output = list(ast.literal_eval(row['output']).values())[0]
    try:
        predict = list(ast.literal_eval(row['predict'].strip()).values())[0]
    except:
        print(row['predict'])
        predict  = ''
    return output,predict
class evaluation:
    def __init__(self):
        """
        初始化方法，可以在这里进行一些初始化操作，比如设置一些默认属性等
        """
        pass


    def handle_ER(self, df):
        """
        处理当task='ER' 且 dataset='wdc' 时的函数逻辑，这里只是一个占位，需根据实际需求完善
        参数:
            df: 传入的数据，示例中假设类似DataFrame结构
        """
        # print(f"Handling task 'ER' with dataset 'wdc' for data: {df}")
        result = df
        result.columns = ['instruction', 'input', 'output', 'prediction']
        result_output = result.apply(Transfer,axis=1,result_type='expand')
        # 这里可以添加实际对数据的处理逻辑
        print('Precision:{}\n\nRecall:{}\n\nF1:{}'.format(
                                                          precision_score(y_true=result_output[0],y_pred=result_output[1]),recall_score(y_true=result_output[0],y_pred=result_output[1]),f1_score(y_true=result_output[0],y_pred=result_output[1])))
        metrics = {}
        metrics['precision'] = precision_score(y_true=result_output[0],y_pred=result_output[1])
        metrics['recall'] = recall_score(y_true=result_output[0],y_pred=result_output[1])
        metrics['F1'] = f1_score(y_true=result_output[0],y_pred=result_output[1])
        return metrics
    def handle_CTA_SimTab(self,df):
        all_metrics = {}
        SimTab_test = df
        # SimTab_test['index'] = SimTab_test.index    
        SimTab_test_Transform = SimTab_test.apply(Ast,axis=1,result_type='expand')    
        all_relation = np.load('train/CTA/SimTab/sim_all_relation.npy')
        SimTab_test_Transform.columns = ['truth','pred','index']
        count = 0
        truth_list = []
        pred_list = []
        for i in range(len(SimTab_test_Transform['index'].unique())):
            select_df = SimTab_test_Transform[SimTab_test_Transform['index'] == i]
            truth = select_df.iloc[0,0]
            select_df_filter = select_df[select_df['pred'].isin(all_relation)]
            try:
                pred = select_df_filter['pred'].value_counts().idxmax()
            except:
                pred = select_df['pred'].value_counts().idxmax()
            truth_list.append(truth)
            pred_list.append(pred)
            if truth==pred:
                count += 1
        relation_dict = {}
        # all_relation = list(SimTab_test_Transform['truth'].unique())
        # all_relation = np.load('data/CTA/sim_all_relation.npy')
        # all_relation.extend(list(SimTab_test_Transform['pred'].unique()))
        for i in range(len(all_relation)):
            relation_dict[all_relation[i]] = i
        SimTab_F1 = pd.DataFrame()
        SimTab_F1['pred_output'] = pred_list
        SimTab_F1['truth_output'] = truth_list
        from sklearn.metrics import f1_score
        pred = SimTab_F1['pred_output'].map(relation_dict).fillna(0).to_list()
        truth = SimTab_F1['truth_output'].map(relation_dict).to_list()
        print('Micro Score: {}\n\nMacro Score: {}'.format(f1_score(y_pred=pred,y_true=truth,average='micro'),
                                                          f1_score(y_pred=pred,y_true=truth,average='macro'))
        )
        all_metrics['micro_f1'] = f1_score(y_pred=pred,y_true=truth,average='micro')
        all_metrics['macro_f1'] = f1_score(y_pred=pred,y_true=truth,average='macro')
        return all_metrics
    def handle_CTA_WebTable(self,df):
        all_metric = {}
        SimTab_test = df
        # SimTab_test['index'] = SimTab_test.index    
        SimTab_test_Transform = SimTab_test.apply(Ast,axis=1,result_type='expand')    
        all_relation = np.load('train/CTA/WebTable/webtable_all_relation.npy')
        SimTab_test_Transform.columns = ['truth','pred','index']
        count = 0
        truth_list = []
        pred_list = []
        for i in range(len(SimTab_test_Transform['index'].unique())):
            select_df = SimTab_test_Transform[SimTab_test_Transform['index'] == i]
            truth = select_df.iloc[0,0]
            select_df_filter = select_df[select_df['pred'].isin(all_relation)]
            try:
                pred = select_df_filter['pred'].value_counts().idxmax()
            except:
                pred = select_df['pred'].value_counts().idxmax()
            truth_list.append(truth)
            pred_list.append(pred)
            if truth==pred:
                count += 1
        relation_dict = {}
        # all_relation = list(SimTab_test_Transform['truth'].unique())
        # all_relation = np.load('data/CTA/sim_all_relation.npy')
        # all_relation.extend(list(SimTab_test_Transform['pred'].unique()))
        for i in range(len(all_relation)):
            relation_dict[all_relation[i]] = i
        SimTab_F1 = pd.DataFrame()
        SimTab_F1['pred_output'] = pred_list
        SimTab_F1['truth_output'] = truth_list
        from sklearn.metrics import f1_score
        pred = SimTab_F1['pred_output'].map(relation_dict).fillna(0).to_list()
        truth = SimTab_F1['truth_output'].map(relation_dict).to_list()
        print('Micro Score: {}\n\nMacro Score: {}'.format(f1_score(y_pred=pred,y_true=truth,average='micro'),
                                                          f1_score(y_pred=pred,y_true=truth,average='macro'))
        )
        all_metric['micro_f1'] = f1_score(y_pred=pred,y_true=truth,average='micro')
        all_metric['macro_f1'] = f1_score(y_pred=pred,y_true=truth,average='macro')
        return all_metric
    def handle_RE_RE(self,df):
        result_metric = {}
        SimTab_test = df
        SimTab_test['index'] = SimTab_test.index    
        SimTab_test_Transform = SimTab_test.apply(Ast,axis=1,result_type='expand')    
        all_relation = np.load('train/RE/all_relation.npy')
        ground_truth = pd.read_csv('train/RE/test_RAG.csv',index_col=0)
        SimTab_test_Transform.columns = ['truth','pred','index']
        count = 0
        truth_list = []
        pred_list = []
        for i in range(len(SimTab_test_Transform['index'].unique())):
            select_df = SimTab_test_Transform[SimTab_test_Transform['index'] == i]
            # truth = select_df.iloc[0,0]
            truth = eval(ground_truth.iloc[i,1])
            select_df_filter = select_df[select_df['pred'].isin(all_relation)]
            try:
                pred = select_df_filter['pred'].value_counts().idxmax()
            except:
                pred = select_df['pred'].value_counts().idxmax()
            truth_list.append(truth)
            pred_list.append(pred)
            if truth.__contains__(pred):
                count += 1
        # relation_dict = {}
        # all_relation = list(SimTab_test_Transform['truth'].unique())
        # all_relation = np.load('data/CTA/sim_all_relation.npy')
        # all_relation.extend(list(SimTab_test_Transform['pred'].unique()))
        # for i in range(len(all_relation)):
        #     relation_dict[all_relation[i]] = i
        # SimTab_F1 = pd.DataFrame()
        # SimTab_F1['pred_output'] = pred_list
        # SimTab_F1['truth_output'] = truth_list
        # from sklearn.metrics import f1_score
        # pred = SimTab_F1['pred_output'].map(relation_dict).fillna(0).to_list()
        # truth = SimTab_F1['truth_output'].map(relation_dict).to_list()
        # print('Acc(Coverage): {}'.format(
        #     count / len(SimTab_test_Transform['index'].unique())
        # )
        # )
        result_metric['acc'] = count / len(SimTab_test_Transform['index'].unique())
        f1_metric = calculate_f1_scores(pred_list=pred_list,truth_list=truth_list)
        for key in f1_metric:
            result_metric[key] = f1_metric[key]
        return result_metric
    def handle_DC_rayyan(self,df):
        result = df
        rayyan_detector = np.load('train/DC/rayyan/detector.npy')
        rayyan_clean = pd.read_csv('train/DC/rayyan/clean.csv').fillna('')
        rayyan_dirty = pd.read_csv('train/DC/rayyan/dirty.csv').fillna('')
        rayyan_clean = rayyan_clean.apply(Str2Int,axis=1)
        rayyan_dirty = rayyan_dirty.apply(Str2Int,axis=1)
        count = 0
        valid_count = 0
        rayyan_correction = rayyan_dirty.copy()
        import ast
        for d in tqdm(np.argwhere(rayyan_detector==1)):
            i = d[0]
            j = d[1] + 1 ## Ignore Index
            try:
                predict = list(eval(result.iloc[count,-1]).values())[0]
                rayyan_correction.iloc[i,j] = predict
                valid_count += 1
            except:
                # print(result.iloc[count,-1])
                predict = result.iloc[count,-1]
                rayyan_correction.iloc[i,j] = predict
            count += 1
        All_Data_Error = 0
        All_Fixed_Error = 0
        Correct_Fixed_Error = 0
        clean = rayyan_clean.copy()
        dirty = rayyan_dirty.copy()
        correction = rayyan_correction.copy()
        for i in tqdm(range(len(clean))):
        # for i in tqdm(tax_error):
            for j in range(clean.shape[1]):
                dirty_cell = dirty.iloc[i,j]
                clean_cell = clean.iloc[i,j]
                correct_cell = correction.iloc[i,j]
                if(correct_cell!=dirty_cell):
                    All_Fixed_Error += 1
                if(clean_cell!=dirty_cell):
                    All_Data_Error += 1
                    if(correct_cell==clean_cell or correct_cell in clean_cell):
                        Correct_Fixed_Error += 1
        Precision_hospital = Correct_Fixed_Error / All_Fixed_Error
        Recall_hospital = Correct_Fixed_Error / All_Data_Error
        F1_hospital = (2 * Precision_hospital * Recall_hospital) / (Precision_hospital + Recall_hospital)
        print('Precision: {}\n\nRecall: {}\n\nF1 Score: {}'.format(
        Precision_hospital,Recall_hospital,F1_hospital)
        )
        metrics = {}
        metrics['Precision'] = Precision_hospital
        metrics['Recall'] = Recall_hospital
        metrics['F1'] = F1_hospital
        return metrics
    def handle_DC_beer(self,df):
        beer_result = df
        count = 0
        beer_clean = pd.read_csv('train/DC/beer/clean.csv').fillna('')
        beer_dirty = pd.read_csv('train/DC/beer/dirty.csv').fillna('')
        detector_beer = np.load('train/DC/beer/detector.npy')
        beer_dirty.columns = beer_clean.columns
        beer_clean = beer_clean.apply(try_convert_to_int,axis=1).astype(str)
        beer_dirty = beer_dirty.apply(try_convert_to_int,axis=1).astype(str)
        beer_correction = beer_dirty.copy()
        for d in np.argwhere(detector_beer==1):
            i = d[0] 
            j = d[1] + 2
            
            try:
                predict = list(eval(beer_result.iloc[count,-1]).values())[0]
                beer_correction.iloc[i,j] = predict
                count += 1
            except:
                print(count)
                count += 1
        print(len(df),count,detector_beer.sum())
        assert len(df)==count
        All_Data_Error = 0
        All_Fixed_Error = 0
        Correct_Fixed_Error = 0
        print(count,len(np.argwhere(detector_beer==1)),len(beer_result))
        for i in range(len(beer_clean)):
            for j in range(11):
                dirty_cell = beer_dirty.iloc[i,j]
                clean_cell = beer_clean.iloc[i,j]
                correct_cell = beer_correction.iloc[i,j]
                if(correct_cell!=dirty_cell):
                    All_Fixed_Error += 1
                if(clean_cell!=dirty_cell):
                    All_Data_Error += 1
                    if(correct_cell==clean_cell):
                        Correct_Fixed_Error += 1
        Precision_hospital = Correct_Fixed_Error / All_Fixed_Error
        Recall_hospital = Correct_Fixed_Error / All_Data_Error
        F1_hospital = (2 * Precision_hospital * Recall_hospital) / (Precision_hospital + Recall_hospital)
        print('Precision: {}\n\nRecall: {}\n\nF1 Score: {}'.format(
        Precision_hospital,Recall_hospital,F1_hospital)
        )
        metrics = {}
        metrics['Precision'] = Precision_hospital
        metrics['Recall'] = Recall_hospital
        metrics['F1'] = F1_hospital
        return metrics
    def handle_DC_hospital(self,df):
        # hospital_result = pd.read_csv('/data/home/wangys/LLaMA-Factory-main/inference_multi_experts/merge_experts_hospital|Expert-8/mistral-7b-hospital-test.csv',index_col=0) ## 
        hospital_result = df
        count = 0
        hospital_clean = pd.read_csv('train/DC/hospital/clean.csv').astype(str)
        hospital_dirty = pd.read_csv('train/DC/hospital/dirty.csv').astype(str)
        hospital_dirty.columns = hospital_clean.columns
        hospital_correction = hospital_dirty.copy()
        hospital_detector = np.load('train/DC/hospital/detector.npy').reshape((-1,20))
        import ast
        for d in np.argwhere(hospital_detector==1):
            i = d[0]
            j = d[1]
            try:
                predict = list(eval(hospital_result.iloc[count,-1]).values())[0]
                hospital_correction.iloc[i,j] = predict
                count += 1
            except:
                predict = hospital_result.iloc[count,-1]
                hospital_correction.iloc[i,j] = predict
                count += 1
        print(count,len(hospital_result))
        All_Data_Error = 0
        All_Fixed_Error = 0
        Correct_Fixed_Error = 0
        clean = hospital_clean.copy()
        dirty = hospital_dirty.copy()
        correction = hospital_correction.copy()
        for i in tqdm(range(len(clean))):
        # for i in tqdm(tax_error):
            for j in range(clean.shape[1]):
                dirty_cell = dirty.iloc[i,j]
                clean_cell = clean.iloc[i,j]
                correct_cell = correction.iloc[i,j]
                if(correct_cell!=dirty_cell):
                    All_Fixed_Error += 1
                if(clean_cell!=dirty_cell):
                    All_Data_Error += 1
                    if(correct_cell==clean_cell or correct_cell in clean_cell):
                        Correct_Fixed_Error += 1
        Precision_hospital = Correct_Fixed_Error / All_Fixed_Error
        Recall_hospital = Correct_Fixed_Error / All_Data_Error
        F1_hospital = (2 * Precision_hospital * Recall_hospital) / (Precision_hospital + Recall_hospital)
        Precision_hospital,Recall_hospital,F1_hospital
        print('Precision: {}\n\nRecall: {}\n\nF1 Score: {}'.format(
        Precision_hospital,Recall_hospital,F1_hospital)
        )
        metrics = {}
        metrics['Precision'] = Precision_hospital
        metrics['Recall'] = Recall_hospital
        metrics['F1'] = F1_hospital
        return metrics
    def handle_imputation(self,df):
        walmart_test = df
        walmart_DI_ast = walmart_test.apply(Impuration_AST,axis=1,result_type='expand')
        Acc = 1 - len(walmart_DI_ast[walmart_DI_ast[0]!=walmart_DI_ast[1]]) / len(walmart_DI_ast)
        print('Accuracy:{}'.format(Acc))
    def handle_imputation_restaurant(self,df):
        walmart_test = df
        walmart_test['output'] = walmart_test['output'].str.replace("` new york city '","` new york '")
        walmart_test['output'] = walmart_test['output'].str.replace("` w. hollywood '","hollywood")
        walmart_test['predict'] = walmart_test['predict'].str.replace("` new york city '","` new york '")
        walmart_test['predict'] = walmart_test['predict'].str.replace("` w. hollywood '","hollywood")
        walmart_DI_ast = walmart_test.apply(Impuration_AST,axis=1,result_type='expand')
        Acc = 1 - len(walmart_DI_ast[walmart_DI_ast[0]!=walmart_DI_ast[1]]) / len(walmart_DI_ast)
        print('Accuracy:{}'.format(Acc))
        metrics = {}
        metrics['Accuray'] = Acc
        return metrics
    def handle_AVE(self,df):
        ave_train = df
        ave_train = ave_train.apply(AVE_AST,axis=1)
        Acc = 1 - len(ave_train[ave_train['output_item']!=ave_train['predict_item']]) / len(ave_train)
        print('Accuracy:{}'.format(Acc))
        metrics = {}
        metrics['Accuray'] = Acc
        return metrics
    def handle_other_task_dataset(self, task, dataset, df):
        """
        处理其他任务和数据集组合的情况，这里只是简单打印，实际中可按要求定制逻辑
        参数:
            task: 任务名称
            dataset: 数据集名称
            df: 传入的数据
        """
        print(f"Handling other task '{task}' and dataset '{dataset}' for data: {df}")

    def process(self, task, dataset, file):
        """
        根据传入的任务、数据集和文件来决定调用哪个具体函数进行处理
        参数:
            task: 任务名称
            dataset: 数据集名称
            file: 数据文件（示例中类似DataFrame的结构）
        """
        if task == 'ER' or task == 'SM':
            metrics = self.handle_ER(file)
        elif task =='CTA' and dataset=='SimTab':
            metrics = self.handle_CTA_SimTab(file)
        elif task =='CTA' and dataset=='WebTable':
            metrics = self.handle_CTA_WebTable(file)
        elif task =='RE' and dataset=='RE':
            metrics = self.handle_RE_RE(file)
        elif task =='DC' and dataset=='rayyan':
            metrics = self.handle_DC_rayyan(file)
        elif task =='DC' and dataset=='beer':
            metrics = self.handle_DC_beer(file)
        elif task =='DC' and dataset=='hospital':
            metrics = self.handle_DC_hospital(file)
        elif task=='DI' and dataset=='restaurant':
            metrics = self.handle_imputation_restaurant(file)
        elif task=='DI':
            metrics = self.handle_imputation(file)
        elif task=='AVE':
            metrics = self.handle_AVE(file)
        return metrics
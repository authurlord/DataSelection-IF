import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import evaluate
from torch.cuda.amp import autocast, GradScaler
def compute_gradient_iterative_inner(model, train_dataloader_stochastic,val_dataloader_stochastic,device):
    # Compute the gradient
    # self.model.to(self.device)
    # self.model = torch.compile(self.model,mode="max-autotune")
    # self.model.eval()
    model = model.eval()
    tr_grad_dict = {}
    for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
        grad_dict = {}
        grad_dict['ids'] = batch['id']
        batch.pop('id', None)
        
        
        model.zero_grad()  # zeroing out gradient
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        
        # print(self.model.named_parameters)
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
                # grad_dict[k] = v.grad
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
                # grad_dict[k] = v.grad.T
            elif 'modules_to_save.default.weight' in k:
                grad_dict[k] = v.grad.cpu()
                # grad_dict[k] = v.grad
        tr_grad_dict[step] = grad_dict
        del grad_dict
        
    val_grad_dict = {}
    for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
        grad_dict = {}
        grad_dict['ids'] = batch['id']
        batch.pop('id', None)
        
        
        model.zero_grad()  # zeroing out gradient
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        
        for k, v in model.named_parameters():
            # print(k)
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            elif 'modules_to_save.default.weight' in k:
                grad_dict[k] = v.grad.cpu()
        val_grad_dict[step] = grad_dict    
        del grad_dict
        
    return tr_grad_dict, val_grad_dict

class LORAEngineDebertaMultiClass(object):
    def __init__(self,
                 model_name_or_path="microsoft/deberta-v3-base",
                 target_modules=None,
                 train_dataloader=None,
                 eval_dataloader=None,
                 test_dataloader=None,
                 device="cuda",
                 num_epochs=10,
                 lr=3e-4,
                 low_rank=2,
                 task="mrpc",
                 save_path=None,
                 valid_each_epoch=False,
                 cal_grad_per_sample = False,
                 tokenized_dataset = None,
                 grad_epoch = None):
        self.model_name_or_path = model_name_or_path
        self.target_modules = target_modules
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.cal_grad_per_sample = cal_grad_per_sample
        self.num_epochs = num_epochs
        self.lr = lr
        self.task = task
        self.valid_each_epoch = valid_each_epoch
        self.low_rank = low_rank
        # self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.save_path = save_path
        self.grad_epoch = grad_epoch
        ## 如果不指定，只求最后一个epoch的grad返回
        if self.grad_epoch==None:
            self.grad_epoch = [self.num_epochs-1]
        # self.tokenized_dataset = tokenized_dataset

    def build_LORA_model(self):
        # self.model = DebertaV2ForSequenceClassification.from_pretrained(self.model_name_or_path,
        #                                                                 return_dict=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True,compile_model=False)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 inference_mode=False,
                                 target_modules=self.target_modules,
                                 r=self.low_rank,
                                 lora_alpha=self.low_rank,
                                 lora_dropout=0.05,
                                 use_rslora=False)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    def load(self, path):
        """
        从指定路径加载已保存的LoRA模型及相关配置，并设置为当前模型。
        """
        # self.model = PeftModel.from_pretrained(DebertaV2ForSequenceClassification.from_pretrained(
        #     self.model_name_or_path, return_dict=True), path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path,
                                                                        return_dict=True)
        self.model.to(self.device)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(path)
    def load_pretrained_network(self,adapter_path):
        # setup tokenizer
        peft_config = PeftConfig.from_pretrained(adapter_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path,
                                                                        return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        # self.model.config.use_cache = False
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # self.model.config.eos_token_id = self.tokenizer.eos_token_id

        # load a base model


        # load a pre-trained model.
        self.model = PeftModel.from_pretrained(self.model, adapter_path, config=peft_config,is_trainable=True)
        # self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=adapter_path)
    def save_full(self, path):
        """
        保存训练好的LoRA模型、分词器以及相关配置到指定路径。
        """
        self.model = self.model.merge_and_unload() ## 存储全量模型
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    def save_lora(self, path, full_model = False):
        """
        保存训练好的LoRA模型、分词器以及相关配置到指定路径。
        """
        if full_model:

            self.model = self.model.merge_and_unload() ## 存储全量模型
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # torch.save(self.model.state_dict(), path) ## torch原生方法测试
    def compute_gradient_iterative_inner(self, train_dataloader_stochastic,val_dataloader_stochastic):
        # Compute the gradient
        # self.model.to(self.device)
        # self.model = torch.compile(self.model,mode="max-autotune")
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            grad_dict = {}
            grad_dict['ids'] = batch['id']
            batch.pop('id', None)
            
            
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            
            # print(self.model.named_parameters)
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                    # grad_dict[k] = v.grad
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                    # grad_dict[k] = v.grad.T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
                    # grad_dict[k] = v.grad
            tr_grad_dict[step] = grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            grad_dict = {}
            grad_dict['ids'] = batch['id']
            batch.pop('id', None)
            
            
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            
            for k, v in self.model.named_parameters():
                # print(k)
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
            val_grad_dict[step] = grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict
    def train_LORA_model(self):
        """
        执行模型训练的方法，加入混合精度训练相关逻辑，包括加载评估指标、初始化优化器和学习率调度器，
        按轮次遍历训练数据进行训练，并在验证集上进行评估（如果设置了相应标志），
        最后在测试集上进行评估。
        """
        
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (len(self.train_dataloader) * self.num_epochs),
            num_training_steps=(len(self.train_dataloader) * self.num_epochs),
        )
        print('Total Steps:{}'.format(len(self.train_dataloader) * self.num_epochs))

        self.model.to(self.device)

        scaler = GradScaler()  # 创建梯度缩放器，用于混合精度训练
        tr_grad_dict_all = {}
        val_grad_dict_all = {}
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.pop('id', None)
                batch = batch.to(self.device)
                with autocast():  # 使用自动混合精度上下文
                    outputs = self.model(**batch)
                    loss = outputs.loss
                total_loss += loss.item()
                scaler.scale(loss).backward()  # 缩放损失后进行反向传播
                scaler.step(optimizer)  # 通过缩放器执行优化器步骤
                scaler.update()  # 更新缩放器状态
                lr_scheduler.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1}: Training Loss = {total_loss / len(self.train_dataloader)}")

            if self.valid_each_epoch:
                metric = evaluate.load("../../evaluate-main/metrics/f1", "f1")
                self.model.eval()
                for step, batch in enumerate(tqdm(self.eval_dataloader)):
                    batch.pop('id', None)
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    predictions, references = predictions, batch["labels"]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                eval_metric = metric.compute()
                print(f"Epoch {(epoch + 1)}:", eval_metric)
            if self.cal_grad_per_sample and epoch in self.grad_epoch:
                tr_grad_dict,val_grad_dict = compute_gradient_iterative_inner(self.model, 
                                                                        self.train_dataloader, 
                                                                        self.eval_dataloader,
                                                                        device=self.device)
                tr_grad_dict_all[epoch] = tr_grad_dict
                val_grad_dict_all[epoch] = val_grad_dict
                
        self.model.eval()
        metric = evaluate.load("../../evaluate-main/metrics/f1", "f1")
        for step, batch in enumerate(tqdm(self.test_dataloader)):
            batch.pop('id', None)
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print("Prediction Result on Test Data:", eval_metric)
        return tr_grad_dict_all,val_grad_dict_all
    def eval_LORA_model(self):
        """
        执行模型训练的方法，加入混合精度训练相关逻辑，包括加载评估指标、初始化优化器和学习率调度器，
        按轮次遍历训练数据进行训练，并在验证集上进行评估（如果设置了相应标志），
        最后在测试集上进行评估。
        """
        self.model.to(self.device)
        self.model.eval()
        metric = evaluate.load("../../evaluate-main/metrics/f1", "f1")
        # optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # # Instantiate scheduler
        # lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=0.06 * (len(self.train_dataloader) * self.num_epochs),
        #     num_training_steps=(len(self.train_dataloader) * self.num_epochs),
        # )
        # # print('Total Steps:{}'.format(len(self.train_dataloader) * self.num_epochs))

        # # self.model.to(self.device)
        
        # scaler = GradScaler()  # 创建梯度缩放器，用于混合精度训练

        # for epoch in range(self.num_epochs):
        #     self.model.train()
        #     total_loss = 0

        #     for step, batch in enumerate(tqdm(self.train_dataloader)):
        #         batch = batch.to(self.device)
        #         with autocast():  # 使用自动混合精度上下文
        #             outputs = self.model(**batch)
        #             loss = outputs.loss
        #         total_loss += loss.item()
        #         scaler.scale(loss).backward()  # 缩放损失后进行反向传播
        #         scaler.step(optimizer)  # 通过缩放器执行优化器步骤
        #         scaler.update()  # 更新缩放器状态
        #         lr_scheduler.step()
        #         optimizer.zero_grad()

        #     print(f"Epoch {epoch + 1}: Training Loss = {total_loss / len(self.train_dataloader)}")

        #     if self.valid_each_epoch:
        #         self.model.eval()
        #         for step, batch in enumerate(tqdm(self.eval_dataloader)):
        #             batch = batch.to(self.device)
        #             with torch.no_grad():
        #                 outputs = self.model(**batch)
        #             predictions = outputs.logits.argmax(dim=-1)
        #             predictions, references = predictions, batch["labels"]
        #             metric.add_batch(
        #                 predictions=predictions,
        #                 references=references,
        #             )

        #         eval_metric = metric.compute()
        #         print(f"Epoch {(epoch + 1)}:", eval_metric)
        
        for step, batch in enumerate(tqdm(self.test_dataloader)):
            
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print("Prediction Result on Test Data:", eval_metric)

    def compute_gradient(self, tokenized_datasets, collate_fn, batch_size=1):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"],
                                                shuffle=False,
                                                collate_fn=collate_fn,
                                                batch_size=batch_size)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"],
                                            shuffle=False,
                                            collate_fn=collate_fn,
                                            batch_size=batch_size)

        self.model.to(self.device)
        self.model.eval()

        # 提取模型的参数和缓冲区，用于后续的functional操作（类似参考代码的处理）
        func_weights = dict(self.model.named_parameters())
        func_buffers = dict(self.model.named_buffers())

        # 定义计算单个样本梯度的函数
        def single_sample_grad(sample_tensors, model, weights, buffers):
            input_ids, attention_mask = sample_tensors

            def model_output(_model, _weights, _buffers, _input_ids, _attention_mask):
                sample_dict = {
                    "input_ids": _input_ids.unsqueeze(0),  # 增加batch维度，因为模型期望的输入是有batch维度的
                    "attention_mask": _attention_mask.unsqueeze(0)
                }
                return _model(**sample_dict).loss

            grads_loss = torch.func.grad(
                model_output, has_aux=False, argnums=1
            )
            return grads_loss(model, weights, buffers, input_ids, attention_mask)

        # 计算训练集的样本梯度
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sample_tensors = [input_ids, attention_mask]
            # 使用vmap对单个样本梯度计算函数在批次维度向量化
            # grads = torch.func.vmap(
            #     single_sample_grad,
            #     in_dims=(0, None, None, None),
            #     randomness="different"
            # )(sample_tensors, self.model, func_weights, func_buffers)
            grads = torch.func.vmap(
                single_sample_grad,
                in_dims=(None, None, None, *([0] * len(batch))),
                randomness="different",
            )(self.model, func_weights, func_buffers, *batch)

            # 处理梯度字典，筛选并转换梯度格式（和原代码类似的筛选逻辑，根据实际情况调整）
            grad_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = grads[k].cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = grads[k].cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = grads[k].cpu()
            tr_grad_dict[step] = grad_dict
            del grad_dict

        # 计算验证集的样本梯度
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sample_tensors = [input_ids, attention_mask]
            grads = torch.func.vmap(
                single_sample_grad,
                in_dims=(0, None, None, None),
                randomness="different"
            )(sample_tensors, self.model, func_weights, func_buffers)

            grad_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = grads[k].cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = grads[k].cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = grads[k].cpu()
            val_grad_dict[step] = grad_dict
            del grad_dict

        return tr_grad_dict, val_grad_dict
    # def compute_gradient(self, tokenized_datasets, collate_fn, batch_size=1):
    #     train_dataloader_stochastic = DataLoader(tokenized_datasets["train"],
    #                                             shuffle=False,
    #                                             collate_fn=collate_fn,
    #                                             batch_size=batch_size)
    #     val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"],
    #                                         shuffle=False,
    #                                         collate_fn=collate_fn,
    #                                         batch_size=batch_size)

    #     self.model.to(self.device)
    #     self.model.eval()

    #     # 提取模型的参数和缓冲区，用于后续的functional操作（类似参考代码的处理）
    #     func_weights = dict(self.model.named_parameters())
    #     func_buffers = dict(self.model.named_buffers())

    #     # 定义计算单个样本梯度的函数（修改后的逻辑，先计算批次输出和损失，再手动求导）
    #     def single_sample_grad(sample_tensors, model, weights, buffers):
    #         input_ids, attention_mask = sample_tensors

    #         sample_dict = {
    #             "input_ids": input_ids.unsqueeze(0),  # 增加batch维度，因为模型期望的输入是有batch维度的
    #             "attention_mask": attention_mask.unsqueeze(0)
    #         }
    #         # 前向传播得到输出
    #         outputs = model(**sample_dict)
    #         loss = outputs.loss

    #         # 手动求导，这里要注意可能需要处理梯度累积等情况，示例中简单处理
    #         grad_dict = {}
    #         for name, param in weights.items():
    #             if param.grad is not None:
    #                 param.grad.zero_()
    #             loss.backward(retain_graph=True)
    #             grad_dict[name] = param.grad.clone().detach().cpu()

    #         return grad_dict

    #     # 计算训练集的样本梯度
    #     tr_grad_dict = {}
    #     for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
    #         input_ids = batch["input_ids"].to(self.device)
    #         attention_mask = batch["attention_mask"].to(self.device)
    #         sample_tensors = [input_ids, attention_mask]
    #         # 使用vmap对单个样本梯度计算函数在批次维度向量化
    #         grads = torch.func.vmap(
    #             single_sample_grad,
    #             in_dims=(0, None, None, None),
    #             randomness="different"
    #         )(sample_tensors, self.model, func_weights, func_buffers)

    #         # 处理梯度字典，筛选并转换梯度格式（和原代码类似的筛选逻辑，根据实际情况调整）
    #         grad_dict = {}
    #         for k, v in self.model.named_parameters():
    #             if 'lora_A' in k:
    #                 grad_dict[k] = grads[k]
    #             elif 'lora_B' in k:
    #                 grad_dict[k] = grads[k].T
    #             elif 'modules_to_save.default.weight' in k:
    #                 grad_dict[k] = grads[k]
    #         tr_grad_dict[step] = grad_dict
    #         del grad_dict

    #     # 计算验证集的样本梯度
    #     val_grad_dict = {}
    #     for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
    #         input_ids = batch["input_ids"].to(self.device)
    #         attention_mask = batch["attention_mask"].to(self.device)
    #         sample_tensors = [input_ids, attention_mask]
    #         grads = torch.func.vmap(
    #             single_sample_grad,
    #             in_dims=(0, None, None, None),
    #             randomness="different"
    #         )(sample_tensors, self.model, func_weights, func_buffers)

    #         grad_dict = {}
    #         for k, v in self.model.named_parameters():
    #             if 'lora_A' in k:
    #                 grad_dict[k] = grads[k]
    #             elif 'lora_B' in k:
    #                 grad_dict[k] = grads[k].T
    #             elif 'modules_to_save.default.weight' in k:
    #                 grad_dict[k] = grads[k]
    #         val_grad_dict[step] = grad_dict
    #         del grad_dict

    #     return tr_grad_dict, val_grad_dict




    def compute_gradient_iterative(self, tokenized_datasets, collate_fn, batch_size=1, shuffle=False):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn,
                                                  batch_size=batch_size)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                               shuffle=False,
                                               collate_fn=collate_fn,
                                               batch_size=batch_size)
        # Compute the gradient
        # self.model.to(self.device)
        # self.model = torch.compile(self.model,mode="max-autotune")
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            grad_dict = {}
            grad_dict['ids'] = batch['id']
            batch.pop('id', None)
            
            
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            
            # print(self.model.named_parameters)
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                    # grad_dict[k] = v.grad
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                    # grad_dict[k] = v.grad.T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
                    # grad_dict[k] = v.grad
            tr_grad_dict[step] = grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            grad_dict = {}
            grad_dict['ids'] = batch['id']
            batch.pop('id', None)
            
            
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            
            for k, v in self.model.named_parameters():
                # print(k)
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
            val_grad_dict[step] = grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict
    


import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score


class LORAEngineDebertaMultiLabel(object):
    def __init__(self,
                 model_name_or_path="microsoft/deberta-v3-base",
                 use_lora = True,
                 target_modules=None,
                 train_dataloader=None,
                 eval_dataloader=None,
                 test_dataloader=None,
                 device="cuda",
                 num_epochs=10,
                 lr=3e-4,
                 low_rank=2,
                 task="multi_label",
                 save_path=None,
                 valid_each_epoch=False,
                 cal_grad_per_sample=False,
                 tokenized_dataset=None,
                 grad_epoch=None,
                 num_labels = None,
                 label2id = None,
                 id2label = None):
        self.model_name_or_path = model_name_or_path
        self.use_lora = use_lora
        self.target_modules = target_modules
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.cal_grad_per_sample = cal_grad_per_sample
        self.num_epochs = num_epochs
        self.lr = lr
        self.task = task
        self.valid_each_epoch = valid_each_epoch
        self.low_rank = low_rank
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.save_path = save_path
        self.grad_epoch = grad_epoch
        if self.grad_epoch == None:
            self.grad_epoch = [self.num_epochs - 1]
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        # self.tokenized_dataset = tokenized_dataset

    def build_LORA_model(self):
        # 获取训练数据集中的标签映射，用于确定num_labels
        all_labels = []
        for batch in self.train_dataloader:
            labels = batch["labels"].numpy()
            all_labels.extend([l for sublist in labels for l in sublist])
        all_labels = sorted(set(all_labels))
        num_labels = len(all_labels)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            return_dict=True,
            num_labels = self.num_labels,
            id2label = self.id2label,
            label2id = self.label2id,
            problem_type = "multi_label_classification",
            torch_dtype=torch.bfloat16
        )
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if self.use_lora:
            peft_config = LoraConfig(
                task_type="SEQ_CLS",
                inference_mode=False,
                target_modules=self.target_modules,
                r=self.low_rank,
                lora_alpha=self.low_rank,
                lora_dropout=0.05,
                use_rslora=False
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # 保存标签映射，方便后续使用


    def load(self, path):
        """
        从指定路径加载已保存的LoRA模型及相关配置，并设置为当前模型。
        """
        # 加载模型时，尝试获取保存的标签映射信息（假设保存模型时同时保存了相关信息，实际可能需要额外处理）
        self.model = AutoModelForSequenceClassification.from_pretrained(path,
                                                                        return_dict=True,
                                                                        num_labels = self.num_labels,
                                                                        id2label = self.id2label,
                                                                        label2id = self.label2id,
                                                                        problem_type = "multi_label_classification")
        self.model.to(self.device)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # 这里假设从加载的模型相关配置或其他地方能获取到label2id和id2label，需根据实际保存情况调整
        # self.label2id = self.model.config.label2id if hasattr(self.model.config, "label2id") else None
        # self.id2label = self.model.config.id2label if hasattr(self.model.config, "id2label") else None

    def load_pretrained_network(self, adapter_path):
        # setup tokenizer
        peft_config = PeftConfig.from_pretrained(adapter_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            return_dict=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        self.model = PeftModel.from_pretrained(self.model, adapter_path, config=peft_config, is_trainable=True)
        # self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=adapter_path)

    def save_full(self, path):
        """
        保存训练好的LoRA模型、分词器以及相关配置到指定路径。
        """
        self.model = self.model.merge_and_unload()  ## 存储全量模型
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # 同时保存标签映射信息，可以根据实际需求选择合适的保存格式（这里简单示例为保存到模型配置中）
        self.model.config.label2id = self.label2id
        self.model.config.id2label = self.id2label
        self.model.config.save_pretrained(path)

    def save_lora(self, path, full_model=False):
        """
        保存训练好的LoRA模型、分词器以及相关配置到指定路径。
        """
        if full_model:
            self.model = self.model.merge_and_unload()  ## 存储全量模型
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # 同样保存标签映射信息（根据实际保存逻辑调整）
        self.model.config.label2id = self.label2id
        self.model.config.id2label = self.id2label
        self.model.config.save_pretrained(path)


    def train_LORA_model(self):
        """
        执行模型训练的方法，适配多标签分类任务，加入混合精度训练相关逻辑，包括加载评估指标、初始化优化器和学习率调度器，
        按轮次遍历训练数据进行训练，并在验证集上进行评估（如果设置了相应标志），
        最后在测试集上进行评估。
        """
        def compute_metrics(eval_pred):
            clf_metrics = evaluate.combine(["../../evaluate-main/metrics/accuracy",
                                    "../../evaluate-main/metrics/f1",
                                    "../../evaluate-main/metrics/precision",
                                    "../../evaluate-main/metrics/recall"])
            logits, labels = eval_pred
            predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()  # 对logits应用sigmoid并转换为二值预测
            results = clf_metrics.compute(predictions=predictions, references=labels)
            return results
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (len(self.train_dataloader) * self.num_epochs),
            num_training_steps=(len(self.train_dataloader) * self.num_epochs),
        )
        print('Total Steps:{}'.format(len(self.train_dataloader) * self.num_epochs))

        self.model.to(self.device)

        # scaler = GradScaler()  # 创建梯度缩放器，用于混合精度训练

        # 调整评估指标，组合多个评估指标


        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(self.train_dataloader)):
                # batch.pop('id', None)
                # batch = {k: v.to(self.device) for k, v in batch.items()}  # 确保所有张量都在设备上
                batch["labels"] = batch["labels"].float()
                batch = batch.to(self.device)
                # print(batch)
                # with autocast():
                outputs = self.model(**batch)
                logits = outputs.logits
                loss = outputs.loss
                # 确保 labels 是 float 类型，并且形状匹配
                # labels = batch["labels"]
                # if not isinstance(labels, torch.Tensor):
                #     labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
                # else:
                #     labels = labels.float().to(self.device)

                # if logits.shape != labels.shape:
                #     raise ValueError(f"Shape mismatch: Logits {logits.shape}, Labels {labels.shape}")

                # # 计算损失
                # loss_fn = torch.nn.BCEWithLogitsLoss()
                # # print(logits.dtype,labels.dtype,logits,labels)
                # loss = loss_fn(logits, labels)

                total_loss += loss.item()

                loss.backward()                # 直接反向传播
                optimizer.step()               # 更新参数
                lr_scheduler.step()            # 更新学习率调度器
                optimizer.zero_grad()          # 清空梯度

            print(f"Epoch {epoch + 1}: Training Loss = {total_loss / len(self.train_dataloader)}")


            if self.valid_each_epoch:
                all_predictions = []
                all_references = []
                all_logits = []
                self.model.eval()
                # 定义计算评估指标的函数，适配多标签分类任务


                batch.pop('id', None)
                batch["labels"] = batch["labels"].float()
                batch = batch.to(self.device)  # 确保所有张量都在设备上

                with torch.no_grad():
                    outputs = self.model(**batch)
                
                # 使用 sigmoid 将 logits 转为概率，并将其转为二进制预测值
                predictions = (torch.sigmoid(outputs.logits) > 0.5).int()

                # 将预测值和标签转为 CPU 上的 numpy 格式以便后续计算
                all_predictions.append(predictions.cpu().numpy())
                all_references.append(batch["labels"].int().cpu().numpy())

            # 将所有批次的预测值和标签合并
            all_predictions = np.vstack(all_predictions)
            all_references = np.vstack(all_references)

            # 计算多标签分类的 F1、Precision 和 Recall
            precision = precision_score(all_references, all_predictions, average='micro')
            recall = recall_score(all_references, all_predictions, average='micro')
            f1 = f1_score(all_references, all_predictions, average='micro')
            print(f"Epoch {(epoch + 1)}:Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")



        self.model.eval()
        # metric_results = []
        # for step, batch in enumerate(tqdm(self.test_dataloader)):
        #     batch.pop('id', None)
        #     batch["labels"] = batch["labels"].float()
        #     batch = batch.to(self.device)
        #     with torch.no_grad():
        #         outputs = self.model(**batch)
        #     predictions = (torch.sigmoid(outputs.logits) > 0.5).int()
        #     predictions, references = predictions, batch["labels"].int()
        #     metric_results.append(compute_metrics((predictions.cpu().numpy(), references.cpu().numpy())))

        # # 合并每一步的评估指标结果（这里假设指标结果是字典形式，可以根据实际情况调整合并方式）
        # print(metric_results)
        # combined_metrics = {}
        # for result in metric_results:
        #     for key, value in result.items():
        #         combined_metrics[key] = combined_metrics.get(key, 0) + value
        # for key in combined_metrics:
        #     combined_metrics[key] /= len(metric_results)

        # print("Prediction Result on Test Data:", combined_metrics)
        all_predictions = []
        all_references = []
        all_logits = []
        for step, batch in enumerate(tqdm(self.test_dataloader)):
            batch.pop('id', None)
            batch["labels"] = batch["labels"].float()
            batch = batch.to(self.device)  # 确保所有张量都在设备上

            with torch.no_grad():
                outputs = self.model(**batch)
            
            # 使用 sigmoid 将 logits 转为概率，并将其转为二进制预测值
            predictions = (torch.sigmoid(outputs.logits) > 0.5).int()
            all_logits.append(outputs.logits.float().cpu().numpy())
            # 将预测值和标签转为 CPU 上的 numpy 格式以便后续计算
            all_predictions.append(predictions.cpu().numpy())
            all_references.append(batch["labels"].int().cpu().numpy())

        # 将所有批次的预测值和标签合并
        all_predictions = np.vstack(all_predictions)
        all_references = np.vstack(all_references)
        all_logits = np.vstack(all_logits)
        # 计算多标签分类的 F1、Precision 和 Recall
        precision = precision_score(all_references, all_predictions, average='micro')
        recall = recall_score(all_references, all_predictions, average='micro')
        f1 = f1_score(all_references, all_predictions, average='micro')

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        return all_logits
    def eval_LORA_model(self):
        """
        执行模型评估的方法，适配多标签分类任务，包括加载评估指标、遍历测试数据进行评估。
        """
        def compute_metrics(eval_pred):
            clf_metrics = evaluate.combine(["../../evaluate-main/metrics/accuracy",
                                    "../../evaluate-main/metrics/f1",
                                    "../../evaluate-main/metrics/precision",
                                    "../../evaluate-main/metrics/recall"])
            logits, labels = eval_pred
            predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()  # 对logits应用sigmoid并转换为二值预测
            results = clf_metrics.compute(predictions=predictions, references=labels)
            return results
        self.model.to(self.device)
        self.model.eval()

        # 调整评估指标，组合多个评估指标
        clf_metrics = evaluate.combine(["../../evaluate-main/metrics/accuracy",
                                        "../../evaluate-main/metrics/f1",
                                        "../../evaluate-main/metrics/precision",
                                        "../../evaluate-main/metrics/recall"])

        metric_results = []
        for step, batch in enumerate(tqdm(self.test_dataloader)):
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).int()
            predictions, references = predictions, batch["labels"]
            metric_results.append(compute_metrics((outputs.logits.cpu().numpy(), references.cpu().numpy())))

        # 合并每一步的评估指标结果（这里假设指标结果是字典形式，可以根据实际情况调整合并方式）
        combined_metrics = {}
        for result in metric_results:
            for key, value in result.items():
                combined_metrics[key] = combined_metrics.get(key, 0) + value
        for key in combined_metrics:
            combined_metrics[key] /= len(metric_results)

        print("Prediction Result on Test Data:", combined_metrics)




    
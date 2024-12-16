from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.notebook import tqdm
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    DebertaV2ForSequenceClassification, DebertaV2Tokenizer, AdamW, get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate
from transformers import TrainingArguments, Trainer

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class LORAEngine(object):
    def __init__(self, 
                model_name_or_path="roberta-large",
                target_modules=["value"],
                train_dataloader=None,
                eval_dataloader=None,
                device="cuda",
                num_epochs=10,
                lr=3e-4,
                low_rank=2,
                task="mrpc"):
        self.model_name_or_path=model_name_or_path
        self.target_modules=target_modules
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.device=device
        self.num_epochs=num_epochs
        self.lr=lr
        self.task=task
        self.low_rank=low_rank
        
    def build_LORA_model(self):
        '''
        This function fine-tunes a model for classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.model.config.eos_token_id
            
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 inference_mode=False, 
                                 target_modules=self.target_modules,
                                 r=self.low_rank,
                                 lora_alpha=self.low_rank, 
                                 lora_dropout=0.05)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train_LORA_model(self):
        '''
        This function fine-tunes a model for GLUE classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        metric = evaluate.load("../metrics/glue", self.task)
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06*(len(self.train_dataloader)*self.num_epochs),
            num_training_steps=(len(self.train_dataloader)*self.num_epochs),
        )

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.model.eval()
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            print(f"Epoch {(epoch+1)}:", eval_metric)


    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict


class LORAEngineGeneration(object):
    def __init__(self, 
                base_path,
                project_path,
                dataset_name='math_with_reason',
                device="cuda"):
        self.base_path = base_path
        self.project_path = project_path
        self.adapter_path = f"{self.project_path}/models/math_with_reason_13bf"
        self.dataset_name = dataset_name
        self.device=device
        self.load_pretrained_network()
        self.load_datasets()

    def load_pretrained_network(self):
        # setup tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_path)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load a base model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
        base_model = LlamaForCausalLM.from_pretrained(
            self.base_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
            offload_state_dict=True,
        )

        # load a pre-trained model.
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
        self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)

    def load_datasets(self):
        self.train_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_train.hf")
        self.validation_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_test.hf")

    def create_tokenized_datasets(self):
        tokenize_func = lambda x: self.tokenizer(
            x["prompt"], truncation=True, padding=True, max_length=128, return_tensors="pt" # text should be more appropritate
        ).to(self.device)

        if 'with_reason' in self.dataset_name:
            column_list=["text", "answer", "variation", "prompt", "reason"]
        else:
            column_list=["text", "answer", "variation", "prompt"]

        tokenized_datasets=dict()
        tokenized_datasets["train"] = self.train_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        tokenized_datasets["validation"] = self.validation_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")

        return tokenized_datasets, collate_fn

    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            val_grad_dict[step]=grad_dict    
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
                valid_each_epoch=False):
        self.model_name_or_path = model_name_or_path
        # self.target_modules = target_modules or ["query_proj", "key_proj", "value_proj"]  # Typical 
        self.target_modules = target_modules
        # DeBERTa attention layer names
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.task = task
        self.valid_each_epoch = valid_each_epoch
        self.low_rank = low_rank
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name_or_path)
        self.save_path = save_path
    def build_LORA_model(self):
        '''
        This function fine-tunes a DeBERTa model for classification tasks. 
        '''
        self.model = DebertaV2ForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True)
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

    def train_LORA_model(self):
        '''
        This function fine-tunes a DeBERTa model for GLUE classification tasks. 
        '''
        # metric = evaluate.load("/home/yanmy/evaluate-main/metrics/glue", self.task)
        metric = evaluate.load("../metrics/f1", "f1")
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06*(len(self.train_dataloader)*self.num_epochs),
            num_training_steps=(len(self.train_dataloader)*self.num_epochs),
        )
        print('Total Steps:{}'.format(len(self.train_dataloader)*self.num_epochs))
        self.model.to(self.device)
        # self.model = torch.compile(self.model,mode="max-autotune")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss  = 0

            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            print(f"Epoch {epoch + 1}: Training Loss = {total_loss / len(self.train_dataloader)}")
            
            if self.valid_each_epoch:
                self.model.eval()
                for step, batch in enumerate(tqdm(self.eval_dataloader)):
                    batch.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    predictions, references = predictions, batch["labels"]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                eval_metric = metric.compute()
                print(f"Epoch {(epoch+1)}:", eval_metric)
        for step, batch in enumerate(tqdm(self.test_dataloader)):
            batch.to(self.device)
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
        # return self.model
    def save_model(self):
        '''
        保存模型到指定路径
        '''
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")
    def train_LORA_model_multi_gpu(self):
        '''
        This function fine-tunes a DeBERTa model for classification tasks using Hugging Face Trainer. 
        '''
        # Define metric
        metric = evaluate.load("../metrics/f1", "f1")

        # Metric computation function for Trainer
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits.argmax(axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.save_path or "./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_strategy="steps",
            logging_steps=500,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=self.num_epochs,
            learning_rate=self.lr,
            warmup_ratio=0.06,  # Corresponds to 6% of steps for warmup
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",  # Prevent logging to external services
            fp16=True,  # Use mixed precision for faster training
            dataloader_num_workers=4,
            save_steps=500,
            gradient_accumulation_steps=2,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            remove_unused_columns=False,
            push_to_hub=False,
            ddp_find_unused_parameters=False,  # For multi-GPU training
        )

        # Instantiate Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.eval_dataloader.dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        # Start training
        trainer.train()

        # Save the best model to the specified save path
        if self.save_path:
            self.model = trainer.model  # Update self.model with the trained model
            self.model.save_pretrained(self.save_path)
            self.tokenizer.save_pretrained(self.save_path)
        else:
            self.model = trainer.model

    # @staticmethod
    def load(self):
        '''
        从指定路径加载模型
        '''
        path = self.save_path
        self.model = DebertaV2ForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.load_state_dict(torch.load(path))

    def compute_gradient(self, tokenized_datasets, collate_fn, batch_size=1):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
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
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict = {}
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
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict = {}
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
    def compute_gradient_standalone(self, tokenized_datasets, collate_fn, batch_size=1, device='cuda', range = np.arange(0,1000,1)):
        train_dataloader_stochastic = DataLoader(tokenized_datasets, 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=batch_size)
        # val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
        #                                        shuffle=False,
        #                                        collate_fn=collate_fn,
        #                                        batch_size=batch_size)
        # Compute the gradient
        self.model.to(device)
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            if(step in range):
                self.model.zero_grad()  # zeroing out gradient
                batch.to(device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                
                grad_dict = {}
                for k, v in self.model.named_parameters():
                    if 'lora_A' in k:
                        grad_dict[k] = v.grad.cpu()
                    elif 'lora_B' in k:
                        grad_dict[k] = v.grad.cpu().T
                    elif 'modules_to_save.default.weight' in k:
                        grad_dict[k] = v.grad.cpu()
                tr_grad_dict[step] = grad_dict
                del grad_dict
            
        # val_grad_dict = {}
        # for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
        #     self.model.zero_grad()  # zeroing out gradient
        #     batch.to(self.device)
        #     outputs = self.model(**batch)
        #     loss = outputs.loss
        #     loss.backward()
            
        #     grad_dict = {}
        #     for k, v in self.model.named_parameters():
        #         # print(k)
        #         if 'lora_A' in k:
        #             grad_dict[k] = v.grad.cpu()
        #         elif 'lora_B' in k:
        #             grad_dict[k] = v.grad.cpu().T
        #         elif 'modules_to_save.default.weight' in k:
        #             grad_dict[k] = v.grad.cpu()
        #     val_grad_dict[step] = grad_dict    
        #     del grad_dict
            
        return tr_grad_dict
    def compute_gradient_batch(self, tokenized_datasets, collate_fn, batch_size=32):
        """
        Compute per-sample gradients with a batch size > 1.

        Args:
            tokenized_datasets: Dataset dictionary with "train" and "validation".
            collate_fn: Collation function for the dataloader.
            batch_size: Number of samples per batch during computation.

        Returns:
            tr_grad_dict, val_grad_dict: Dictionaries mapping sample indices to gradients.
        """
        # Initialize dataloaders
        train_dataloader_stochastic = DataLoader(
            tokenized_datasets["train"],
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=batch_size
        )
        val_dataloader_stochastic = DataLoader(
            tokenized_datasets["validation"],
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=batch_size
        )

        # Put model in evaluation mode
        self.model.eval()

        # Dictionary to store training gradients
        tr_grad_dict = {}
        train_sample_index = 0

        for step, batch in enumerate(tqdm(train_dataloader_stochastic, desc="Computing Train Gradients")):
            self.model.zero_grad()  # Clear existing gradients
            batch.to(self.device)  # Move batch to device
            outputs = self.model(**batch)
            loss = outputs.loss

            # Compute gradients for the whole batch
            loss.backward(retain_graph=True)

            # Loop through each sample in the batch
            for i in range(batch["input_ids"].size(0)):  # `input_ids` has shape (batch_size, seq_len)
                grad_dict = {}
                for name, param in self.model.named_parameters():
                    if "lora_A" in name:
                        grad_dict[name] = param.grad[i].detach().cpu()
                    elif "lora_B" in name:
                        grad_dict[name] = param.grad[i].detach().cpu().T
                    elif "modules_to_save.default.weight" in name:
                        grad_dict[name] = param.grad[i].detach().cpu()
                # Save gradients for this sample
                tr_grad_dict[train_sample_index] = grad_dict
                train_sample_index += 1

        # Dictionary to store validation gradients
        val_grad_dict = {}
        val_sample_index = 0

        for step, batch in enumerate(tqdm(val_dataloader_stochastic, desc="Computing Validation Gradients")):
            self.model.zero_grad()  # Clear existing gradients
            batch.to(self.device)  # Move batch to device
            outputs = self.model(**batch)
            loss = outputs.loss

            # Compute gradients for the whole batch
            loss.backward(retain_graph=True)

            # Loop through each sample in the batch
            for i in range(batch["input_ids"].size(0)):
                grad_dict = {}
                for name, param in self.model.named_parameters():
                    if "lora_A" in name:
                        grad_dict[name] = param.grad[i].detach().cpu()
                    elif "lora_B" in name:
                        grad_dict[name] = param.grad[i].detach().cpu().T
                    elif "modules_to_save.default.weight" in name:
                        grad_dict[name] = param.grad[i].detach().cpu()
                # Save gradients for this sample
                val_grad_dict[val_sample_index] = grad_dict
                val_sample_index += 1

        return tr_grad_dict, val_grad_dict

    def compute_gradient_ddp(self, tokenized_datasets, collate_fn, batch_size=1):
        # 获取GPU数量
        world_size = torch.cuda.device_count()
        
        # 初始化分布式环境
        dist.init_process_group(backend='nccl')
        
        # 为每个进程设置不同的GPU
        torch.cuda.set_device(self.local_rank)
        
        # 创建分布式数据采样器
        train_sampler = DistributedSampler(tokenized_datasets["train"], num_replicas=world_size, rank=self.local_rank)
        val_sampler = DistributedSampler(tokenized_datasets["validation"], num_replicas=world_size, rank=self.local_rank)
        
        # 创建数据加载器
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                sampler=train_sampler,
                                                collate_fn=collate_fn,
                                                batch_size=batch_size)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                sampler=val_sampler,
                                                collate_fn=collate_fn,
                                                batch_size=batch_size)
        
        # 包装模型以进行分布式数据并行
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # 计算梯度
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad()
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
            tr_grad_dict[step] = grad_dict
            del grad_dict
        
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad()
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
            val_grad_dict[step] = grad_dict    
            del grad_dict
        
        # 清理分布式环境
        dist.destroy_process_group()
        
        return tr_grad_dict, val_grad_dict
    
class LORAEngineDeberta(object):
    def __init__(self, 
                model_name_or_path="microsoft/deberta-v3-base",
                target_modules=None,
                train_dataloader=None,
                eval_dataloader=None,
                device="cuda",
                num_epochs=10,
                lr=3e-4,
                low_rank=2,
                num_labels=2,
                task='mrpc'):  # Set to the number of classes
        self.model_name_or_path = model_name_or_path
        self.target_modules = target_modules
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.low_rank = low_rank
        self.num_labels = num_labels
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name_or_path)
        self.task = task
    def build_LORA_model(self):
        '''
        This function fine-tunes a DeBERTa model for multi-class classification tasks. 
        '''
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,  # Dynamically set the number of labels
            return_dict=True,
        )
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False, 
            target_modules=self.target_modules,
            r=self.low_rank,
            lora_alpha=self.low_rank, 
            lora_dropout=0.05,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train_LORA_model(self):
        '''
        This function fine-tunes a DeBERTa model for multi-class classification tasks using micro F1 metric.
        '''
        metric = evaluate.load("../metrics/f1")  # Load F1 metric
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(0.06 * len(self.train_dataloader) * self.num_epochs),
            num_training_steps=len(self.train_dataloader) * self.num_epochs,
        )

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1}: Training Loss = {total_loss / len(self.train_dataloader)}")

            # Evaluation
            self.model.eval()
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                references = batch["labels"]
                metric.add_batch(
                    predictions=predictions.cpu(),
                    references=references.cpu(),
                )

            # Compute micro F1
            eval_metric = metric.compute(average="micro")
            print(f"Epoch {epoch + 1}: Micro F1 = {eval_metric['f1']}")

    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(
            tokenized_datasets["train"], 
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=32,
        )
        val_dataloader_stochastic = DataLoader(
            tokenized_datasets["validation"], 
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=32,
        )

        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
            tr_grad_dict[step] = grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad()  # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()
            val_grad_dict[step] = grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict
    def compute_gradient_ddp_standalone(model, tokenized_datasets, collate_fn, device="cuda", world_size=4, rank=0):
        """
        Compute gradients on multiple GPUs using PyTorch DistributedDataParallel (DDP).
        
        Args:
            model: Pre-trained model with LoRA layers.
            tokenized_datasets: Dataset dictionary with "train" and "validation" splits.
            collate_fn: Function to collate batches.
            device: Device type ("cuda" or "cpu").
            world_size: Number of GPUs to use.
            rank: Current GPU rank (0 for single-GPU setup).
            
        Returns:
            tr_grad_dict, val_grad_dict: Dictionaries containing gradients for training and validation datasets.
        """
        # Initialize distributed training if running in a multi-GPU environment
        if world_size > 1:
            torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

        # Move model to device and wrap with DDP if applicable
        model.to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)

        # Create distributed samplers for datasets
        train_sampler = DistributedSampler(tokenized_datasets["train"], num_replicas=world_size, rank=rank, shuffle=False)
        val_sampler = DistributedSampler(tokenized_datasets["validation"], num_replicas=world_size, rank=rank, shuffle=False)

        # Create data loaders with distributed samplers
        train_dataloader_stochastic = DataLoader(
            tokenized_datasets["train"],
            sampler=train_sampler,
            collate_fn=collate_fn,
            batch_size=1,
        )
        val_dataloader_stochastic = DataLoader(
            tokenized_datasets["validation"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            batch_size=1,
        )

        # Prepare gradient storage
        tr_grad_dict = {}
        val_grad_dict = {}

        # Put the model in evaluation mode
        model.eval()

        # Train gradient computation
        for step, batch in enumerate(tqdm(train_dataloader_stochastic, desc="Processing Training Data")):
            model.zero_grad()  # Clear existing gradients
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU
            outputs = model(**batch)  # Forward pass
            loss = outputs.loss
            loss.backward()  # Compute gradients

            # Store gradients for train set
            grad_dict = {}
            for k, v in model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()  # Store gradient for lora_A
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T  # Transpose gradient for lora_B
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()  # Store gradient for saved modules
            tr_grad_dict[step] = grad_dict
            del grad_dict  # Free memory

        # Validation gradient computation
        for step, batch in enumerate(tqdm(val_dataloader_stochastic, desc="Processing Validation Data")):
            model.zero_grad()  # Clear existing gradients
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU
            outputs = model(**batch)  # Forward pass
            loss = outputs.loss
            loss.backward()  # Compute gradients

            # Store gradients for validation set
            grad_dict = {}
            for k, v in model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k] = v.grad.cpu()  # Store gradient for lora_A
                elif 'lora_B' in k:
                    grad_dict[k] = v.grad.cpu().T  # Transpose gradient for lora_B
                elif 'modules_to_save.default.weight' in k:
                    grad_dict[k] = v.grad.cpu()  # Store gradient for saved modules
            val_grad_dict[step] = grad_dict
            del grad_dict  # Free memory

        # Cleanup distributed environment if applicable
        if world_size > 1:
            torch.distributed.destroy_process_group()

        return tr_grad_dict, val_grad_dict

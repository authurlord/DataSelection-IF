# Poly (PEFT PolyConfig) multi-task JOINT training: learns a small adapter inventory (n_skills) + a
# per-task routing matrix, so experts are composable BY CONSTRUCTION (training-time coordination ->
# no post-hoc merge conflict). Base = Qwen3.5-4B. FA2 + liger enabled (A800). Routes via task_ids.
import json, glob, argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PolyConfig, get_peft_model, TaskType
from datasets import Dataset

def load_base(path):
    import importlib.util
    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation=attn)
    try:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        m = AutoLigerKernelForCausalLM.from_pretrained(path, **kw); print("LIGER applied")
        return m
    except Exception as e:
        print("liger skip:", str(e)[:100])
        return AutoModelForCausalLM.from_pretrained(path, **kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--data_glob', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--r', type=int, default=16)
    ap.add_argument('--n_skills', type=int, default=4)
    ap.add_argument('--n_splits', type=int, default=4)  # MHR by default; pass --n_splits 1 for plain Poly
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--cutoff', type=int, default=2048)  # swift best-practice max_length
    ap.add_argument('--bsz', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)  # vary for multi-seed error bars
    # swift Qwen3/3.5 best-practice: all-linear targets
    ap.add_argument('--targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--enable_thinking', type=int, default=1)  # 0 for Qwen3.5 (non-thinking SFT)
    a = ap.parse_args()
    files = sorted(glob.glob(a.data_glob))
    tasks = [os.path.basename(f).replace('task-', '').replace('.json', '') for f in files]
    task2id = {t: i for i, t in enumerate(tasks)}
    print('tasks:', task2id)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    rows = []
    for f, t in zip(files, tasks):
        for r in json.load(open(f)):
            instr = r['instruction'] + (('\n' + r['input']) if r.get('input') else '')
            rows.append({'instr': instr, 'resp': str(r['output']), 'task_ids': task2id[t]})
    def tokz(ex):
        msgs = [{'role': 'user', 'content': ex['instr']}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=bool(a.enable_thinking))
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "<|user|>\n%s\n<|assistant|>\n" % ex['instr']
        full = prompt + ex['resp'] + (tok.eos_token or '')
        ids = tok(full, truncation=True, max_length=a.cutoff)['input_ids']
        plen = min(len(tok(prompt, truncation=True, max_length=a.cutoff)['input_ids']), len(ids))
        labels = [-100] * plen + ids[plen:]
        return {'input_ids': ids, 'labels': labels[:len(ids)], 'task_ids': ex['task_ids']}
    ds = Dataset.from_list(rows).map(tokz, remove_columns=['instr', 'resp'])
    print('dataset size', len(ds))
    base = load_base(a.base)
    cfg = PolyConfig(task_type=TaskType.CAUSAL_LM, r=a.r, n_tasks=len(tasks),
                     n_skills=a.n_skills, n_splits=a.n_splits,
                     target_modules=[m.strip() for m in a.targets.split(',')])
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    def collate(feats):
        ml = max(len(f['input_ids']) for f in feats)
        ii, ll, am, ti = [], [], [], []
        for f in feats:
            p = ml - len(f['input_ids'])
            ii.append(f['input_ids'] + [tok.pad_token_id] * p)
            ll.append(f['labels'] + [-100] * p)
            am.append([1] * len(f['input_ids']) + [0] * p)
            ti.append(f['task_ids'])
        return {'input_ids': torch.tensor(ii), 'labels': torch.tensor(ll),
                'attention_mask': torch.tensor(am), 'task_ids': torch.tensor(ti)}
    args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.bsz,
                             seed=a.seed, data_seed=a.seed,
                             gradient_accumulation_steps=8, learning_rate=1e-4, lr_scheduler_type='cosine',
                             warmup_ratio=0.05, logging_steps=20, save_strategy='epoch', bf16=True, report_to=[],
                             gradient_checkpointing=False, dataloader_num_workers=4, remove_unused_columns=False,
                             label_names=["labels", "task_ids"])
    class PolyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            tid = inputs.pop('task_ids', None)
            out = model(**inputs, task_ids=tid)
            loss = out.loss
            return (loss, out) if return_outputs else loss
    PolyTrainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
    model.save_pretrained(a.out)
    json.dump(task2id, open(os.path.join(a.out, 'task2id.json'), 'w'))
    print('POLY TRAIN DONE ->', a.out)

if __name__ == '__main__':
    main()

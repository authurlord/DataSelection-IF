# Exp C (CORRECTED): MELD-DS quality is a TRUNCATED PPL RATIO f(PPL_cond/PPL_prior), NOT a ppl filter.
# Paper Eq.(9): F_qual(S') = sum_q f( PPL_cond(q)/PPL_prior(q) ),  f(x)=x if 0<=x<=1 else 0.
#   PPL_prior = perplexity of label l given query q (NO demonstration)
#   PPL_cond  = perplexity of label l given query q AND retrieved demonstrations q_RAG (k-NN, clean labels)
# Mechanism under label noise: a flipped label is contradicted by the (correct) demonstrations -> conditioning
# RAISES its perplexity -> ratio > 1 -> f()=0 -> the candidate gets ZERO quality score -> down-weighted in the
# submodular argmax (NOT dropped by a threshold). A clean label is helped by demonstrations -> ratio < 1 -> kept.
#
# Proxy model = Qwen2.5-0.5B-Instruct (small, per paper). Retriever = bge-base-en-v1.5. Task = ER amazon-google.
import os, json, re, argparse, ast
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def label_of(out):
    try:
        d = ast.literal_eval(str(out)); v = str(list(d.values())[0]).lower()
        return 'mismatch' if 'mis' in v or 'dis' in v else 'match'
    except Exception:
        return None

def entity_block(instr):
    # the instance content (Entity 1/2), dropping the fixed task template header
    s = instr
    if 'Output format example' in s:
        s = s.split('Output format example', 1)[1]
        s = s.split('\n\n', 1)[1] if '\n\n' in s else s
    return s.strip()[:700]

@torch.no_grad()
def label_ppl(model, tok, prompt, label_str, device):
    # perplexity of label tokens conditioned on prompt (decoder-only, mask the prompt)
    p_ids = tok(prompt, return_tensors='pt', truncation=True, max_length=1600)['input_ids']
    l_ids = tok(label_str, return_tensors='pt', add_special_tokens=False)['input_ids']
    ids = torch.cat([p_ids, l_ids], 1).to(device)
    labels = ids.clone(); labels[0, :p_ids.shape[1]] = -100
    out = model(ids, labels=labels)
    # out.loss is mean NLL over the label tokens
    return float(torch.exp(out.loss).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', default='/home/yanmy/models/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--adapter', default='lora/qwen-0.5B/ER/amazon-google/init',
                    help='warmed proxy M_proxy: per-task init LoRA (S_init). "" = raw model')
    ap.add_argument('--bge', default='/home/yanmy/models/bge-base-en-v1.5')
    ap.add_argument('--data', default='train/ER/amazon-google/train-select.json')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--n', type=int, default=600)   # subsample for speed; -1 = all
    ap.add_argument('--out', default='revision_exp/results/expC_ppl_ratio/er_amazon_google.csv')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    data = json.load(open(a.data))
    data = [d for d in data if label_of(d['output']) in ('match', 'mismatch')]
    if a.n > 0 and len(data) > a.n:
        data = list(np.random.default_rng(0).choice(data, a.n, replace=False))
    instrs = [d['instruction'] for d in data]
    ents = [entity_block(s) for s in instrs]
    true_lab = [label_of(d['output']) for d in data]

    # ---- retrieve k-NN demonstrations (q_RAG) by entity-content similarity, CLEAN labels ----
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer(a.bge, device=a.device)
    emb = bge.encode(ents, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sim = emb @ emb.T; np.fill_diagonal(sim, -1)
    knn = np.argsort(-sim, axis=1)[:, :a.k]
    def demo_block(i):
        lines = []
        for j in knn[i]:
            lines.append('Record pair:\n%s\nJudgement: {"Output": "%s"}' % (ents[j][:400], true_lab[j]))
        return 'Here are similar labeled examples for reference:\n\n' + '\n\n'.join(lines) + '\n\n'

    # ---- proxy model ----
    tok = AutoTokenizer.from_pretrained(a.proxy)
    model = AutoModelForCausalLM.from_pretrained(a.proxy, torch_dtype=torch.float16).to(a.device).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
        print('warmed proxy: loaded init adapter', a.adapter)

    rows = []
    for i in range(len(data)):
        prior_p = instrs[i] + '\n'
        cond_p = demo_block(i) + instrs[i] + '\n'
        rec = {'idx': i, 'true': true_lab[i]}
        for lab in ('match', 'mismatch'):
            ltxt = '{"Output": "%s"}' % lab
            pr = label_ppl(model, tok, prior_p, ltxt, a.device)
            cd = label_ppl(model, tok, cond_p, ltxt, a.device)
            rec['prior_%s' % lab] = round(pr, 4); rec['cond_%s' % lab] = round(cd, 4)
            rec['ratio_%s' % lab] = round(cd / pr, 4)
        rows.append(rec)
        if (i + 1) % 100 == 0: print('%d/%d' % (i + 1, len(data)))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    # ---- analysis: clean-label ratio vs flipped-label ratio + truncation f() ----
    def f(x): return x if (0 <= x <= 1) else 0.0
    flip = {'match': 'mismatch', 'mismatch': 'match'}
    r_clean = np.array([df.loc[i, 'ratio_%s' % df.loc[i, 'true']] for i in range(len(df))])
    r_flip = np.array([df.loc[i, 'ratio_%s' % flip[df.loc[i, 'true']]] for i in range(len(df))])
    fq_clean = np.array([f(x) for x in r_clean]); fq_flip = np.array([f(x) for x in r_flip])
    print('\n=== TRUNCATED PPL RATIO  f(PPL_cond/PPL_prior)  (ER amazon-google, proxy Qwen2.5-0.5B, k=%d) ===' % a.k)
    print('clean label : mean ratio %.3f | median %.3f | %% ratio<=1 (kept, f>0): %.1f%% | mean f-quality %.4f'
          % (r_clean.mean(), np.median(r_clean), 100 * (r_clean <= 1).mean(), fq_clean.mean()))
    print('flipped lbl : mean ratio %.3f | median %.3f | %% ratio>1  (zeroed, f=0): %.1f%% | mean f-quality %.4f'
          % (r_flip.mean(), np.median(r_flip), 100 * (r_flip > 1).mean(), fq_flip.mean()))
    print('separation  : flipped get %.1fx lower mean f-quality than clean; %.1f%% of injected-noise candidates '
          'are zeroed by f() (ratio>1) vs %.1f%% of clean.'
          % ((fq_clean.mean() + 1e-9) / (fq_flip.mean() + 1e-9), 100 * (r_flip > 1).mean(), 100 * (r_clean > 1).mean()))
    json.dump({'k': a.k, 'n': len(df),
               'clean_mean_ratio': float(r_clean.mean()), 'flip_mean_ratio': float(r_flip.mean()),
               'clean_pct_ratio_le1': float(100 * (r_clean <= 1).mean()), 'flip_pct_ratio_gt1': float(100 * (r_flip > 1).mean()),
               'clean_mean_fq': float(fq_clean.mean()), 'flip_mean_fq': float(fq_flip.mean())},
              open(a.out.replace('.csv', '_summary.json'), 'w'), indent=1)
    print('\nSAVED', a.out)

if __name__ == '__main__':
    main()

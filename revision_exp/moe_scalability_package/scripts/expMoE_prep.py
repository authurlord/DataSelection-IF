# Build all data artifacts for the MoE scalability/router experiments (run LOCALLY from repo root).
#   - manifest.csv                          (12 discriminative experts, avoid DC/CTA)
#   - scaling_data/union-n{3,6,9,12}probe.json   (dense joint-FT unions, nested, probe-fixed)
#   - poly_data/n{3,6,9,12}/task-*.json     (Poly per-task files; poly_train.py globs these)
#   - dataset_info_probe.json               (snippet to merge into LLaMA-Factory dataset_info.json on 51.11)
import json, os

# name -> (task, dataset, train_file, test_file)
T = {
 'ER-amazon-google':  ('ER','amazon-google', 'train/ER/amazon-google/train-select.json',  'train/ER/amazon-google/amazon-google-test.json'),
 'ER-walmart-amazon': ('ER','walmart-amazon','train/ER/walmart-amazon/train-select.json', 'train/ER/walmart-amazon/walmart-amazon-test.json'),
 'ER-wdc':            ('ER','wdc',           'train/ER/wdc/train-select.json',            'train/ER/wdc/wdc-test.json'),
 'ER-abt-buy':        ('ER','abt-buy',       'train/ER/abt-buy/train-select.json',        'train/ER/abt-buy/abt-buy-test.json'),
 'ER-semi-text-w':    ('ER','semi-text-w',   'train/ER/semi-text-w/train-select.json',    'train/ER/semi-text-w/semi-text-w-test.json'),
 'ER-semi-text-c':    ('ER','semi-text-c',   'train/ER/semi-text-c/train-select.json',    'train/ER/semi-text-c/semi-text-c-test.json'),
 'DI-amazon':         ('DI','amazon',        'train/DI/amazon/train-select.json',         'train/DI/amazon/amazon-test.json'),
 'DI-walmart':        ('DI','walmart',       'train/DI/walmart/train-select.json',        'train/DI/walmart/walmart-test.json'),
 'SM-CMS':            ('SM','CMS',           'train/SM/CMS/train-select.json',             'train/SM/CMS/CMS-test.json'),
 'SM-synthea':        ('SM','synthea',       'train/SM/synthea/synthea-train.json',        'train/SM/synthea/synthea-test.json'),
 'RE-RE':             ('RE','RE',            'train/RE/RE-train.json',                     'train/RE/RE-test-built.json'),
 'AVE-oa_mine':       ('AVE','oa_mine',      'train/AVE/oa_mine/train-select.json',        'train/AVE/oa_mine/oa_mine-test.json'),
}

# nested, probe-fixed task sets (avoid DC/CTA). probe = first 3 (evaluated at every n).
NESTED = {
 3:  ['ER-amazon-google','DI-amazon','SM-CMS'],
 6:  ['ER-amazon-google','DI-amazon','SM-CMS','ER-walmart-amazon','ER-wdc','RE-RE'],
 9:  ['ER-amazon-google','DI-amazon','SM-CMS','ER-walmart-amazon','ER-wdc','RE-RE','ER-abt-buy','ER-semi-text-w','DI-walmart'],
 12: ['ER-amazon-google','DI-amazon','SM-CMS','ER-walmart-amazon','ER-wdc','RE-RE','ER-abt-buy','ER-semi-text-w','DI-walmart','ER-semi-text-c','SM-synthea','AVE-oa_mine'],
}
PROBE = ['ER-amazon-google','DI-amazon','SM-CMS']

REPO_REMOTE = '/data/yanmengyi/DataSelection-IF'
os.makedirs('revision_exp/scaling_data', exist_ok=True)
os.makedirs('revision_exp/poly_data', exist_ok=True)

# --- manifest.csv (full 12, exp A) ---
with open('revision_exp/manifest.csv', 'w') as f:
    f.write('task,dataset,adapter,test_file\n')
    for name in NESTED[12]:
        task, ds, _tr, te = T[name]
        f.write('%s,%s,lora/mistral-7B/%s/%s/select,%s\n' % (task, ds, task, ds, te))
print('wrote revision_exp/manifest.csv (12 experts)')

# --- dense unions (nested, probe-fixed) ---
def load(name):
    data = json.load(open(T[name][2]))
    norm = []
    for r in data:
        if 'instruction' in r:
            norm.append({'instruction': r['instruction'], 'input': r.get('input', ''), 'output': str(r['output'])})
        else:  # RE-train.json positional schema: '0'=instruction, '1'=input, '2'=output
            norm.append({'instruction': r['0'], 'input': r.get('1', ''), 'output': str(r['2'])})
    return norm

di = {}
for n, names in NESTED.items():
    rows = []
    for name in names:
        rows.extend(load(name))
    out = 'revision_exp/scaling_data/union-n%dprobe.json' % n
    json.dump(rows, open(out, 'w'))
    di['union_n%dprobe' % n] = {'file_name': '%s/%s' % (REPO_REMOTE, out),
                                'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'}}
    print('dense union n=%2d : %5d examples -> %s' % (n, len(rows), out))

json.dump(di, open('revision_exp/dataset_info_probe.json', 'w'), indent=1)
print('wrote revision_exp/dataset_info_probe.json (merge into LLaMA-Factory dataset_info.json)')

# --- Poly per-task files (poly_train.py globs task-*.json; nested via per-n dir) ---
import glob as _glob
for n, names in NESTED.items():
    d = 'revision_exp/poly_data/n%d' % n
    os.makedirs(d, exist_ok=True)
    for stale in _glob.glob(os.path.join(d, 'task-*.json')):   # clear stale -> poly_train glob stays correct
        os.remove(stale)
    for name in names:
        json.dump(load(name), open(os.path.join(d, 'task-%s.json' % name), 'w'))
    print('poly n=%2d : %d task files -> %s' % (n, len(names), d))

print('PROBE tasks (eval at every n):', PROBE)
print('PREP DONE')

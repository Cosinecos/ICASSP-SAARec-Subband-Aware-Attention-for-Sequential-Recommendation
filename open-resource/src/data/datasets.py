import os, json, torch
from torch.utils.data import Dataset
from ..utils.registry import register

PAD = 0
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    if pad_len > 0:
        seq = [PAD] * pad_len + seq
    return seq, pad_len

class SeqDataset(Dataset):
    def __init__(self, jsonl_path, max_len, num_items=None):
        self.max_len = max_len
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.num_items = num_items
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x = self.samples[idx]; seq = x['seq']; tgt = x['target']
        seq_pad, n_pad = pad_seq(seq, self.max_len)
        attn_mask = [0]*n_pad + [1]*(self.max_len - n_pad)
        return {
            'user': x.get('user', 0),
            'seq': torch.tensor(seq_pad, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.float32),
            'target': torch.tensor(tgt, dtype=torch.long)
        }

def build_generic(data_dir, max_len):
    def p(n): return os.path.join(data_dir, f'{n}.jsonl')
    meta = json.load(open(os.path.join(data_dir, 'meta.json'), 'r', encoding='utf-8'))
    train = SeqDataset(p('train'), max_len, meta.get('num_items'))
    val   = SeqDataset(p('val'),   max_len, meta.get('num_items'))
    test  = SeqDataset(p('test'),  max_len, meta.get('num_items'))
    return train, val, test, meta

@register('dataset', 'ml1m')
def build_ml1m(data_dir, max_len): return build_generic(data_dir, max_len)

@register('dataset', 'amazon_beauty')
def build_beauty(data_dir, max_len): return build_generic(data_dir, max_len)

@register('dataset', 'amazon_sports')
def build_sports(data_dir, max_len): return build_generic(data_dir, max_len)

@register('dataset', 'mind_small')
def build_mind_small(data_dir, max_len): return build_generic(data_dir, max_len)

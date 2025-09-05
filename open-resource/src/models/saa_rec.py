import math, torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_wavelets as ptwt
from ..utils.registry import register

class TFLN(nn.Module):
    def __init__(self, d, eps=1e-12):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = (x-mu).pow(2).mean(-1, keepdim=True)
        return (x-mu)/torch.sqrt(var+self.eps)*self.g + self.b

class FFN(nn.Module):
    def __init__(self, d, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d, 4*d)
        self.fc2 = nn.Linear(4*d, d)
        self.do  = nn.Dropout(p)
        self.ln  = TFLN(d)
    def forward(self, x):
        y = self.fc1(x)
        y = y * 0.5 * (1.0 + torch.erf(y / math.sqrt(2.0)))
        y = self.fc2(y)
        y = self.do(y)
        return self.ln(y + x)

class TinyBandAttn(nn.Module):
    def __init__(self, d, p=0.2):
        super().__init__()
        self.q = nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d)
        self.o = nn.Linear(d,d);  self.do=nn.Dropout(p); self.ln=TFLN(d)
        self.scale = d**0.5
    def forward(self, x):
        q,k,v = self.q(x), self.k(x), self.v(x)
        a = torch.matmul(q, k.transpose(-1,-2))/self.scale
        a = F.softmax(a, dim=-1); a=self.do(a)
        y = torch.matmul(a, v); y=self.o(y); y=self.do(y)
        return self.ln(y + x)

def band_energy(x): return x.pow(2).mean(dim=(1,2))
def band_sfm(x, eps=1e-8):
    p = x.pow(2) + eps
    geo = torch.exp(torch.log(p).mean(dim=(1,2)))
    ari = p.mean(dim=(1,2))
    return geo/(ari+eps)

class BoundaryTokens(nn.Module):
    def __init__(self, d, L=4, reflect=False):
        super().__init__()
        self.L=L; self.reflect=reflect
        self.l = nn.Parameter(torch.randn(1,L,d)*0.02)
        self.r = nn.Parameter(torch.randn(1,L,d)*0.02)
    def forward(self, x):
        b = x.size(0)
        x = torch.cat([self.l.expand(b,-1,-1), x, self.r.expand(b,-1,-1)], dim=1)
        if self.reflect:
            x = F.pad(x, (0,0,2,2), mode='reflect')
        return x
    def crop(self, x, T):
        if x.size(1) >= T + 2*self.L:
            x = x[:, self.L:self.L+T, :]
        else:
            x = x[:, :T, :]
        return x

class SAASubbandsMixer(nn.Module):
    def __init__(self, d, J=3, wave='db4', p=0.2, L=4, reflect=False, not_restore=False, gate_l1=0.0, gate_entropy=0.0):
        super().__init__()
        self.d=d; self.J=J; self.not_restore=not_restore
        self.do=nn.Dropout(p); self.ln=TFLN(d)
        self.fwd=ptwt.DWT1DForward(wave=wave, J=J)
        self.inv=ptwt.DWT1DInverse(wave=wave)
        self.bd = BoundaryTokens(d, L=L, reflect=reflect)
        self.band_scales = nn.Parameter(torch.randn(J,d)*0.02)
        self.tiny = TinyBandAttn(d, p=p)
        self.sqrt_beta = nn.Parameter(torch.randn(1,1,d)*0.02)
        self.gate_l1=gate_l1; self.gate_entropy=gate_entropy
        gate_in = 2 + J; hid=max(8,4*J)
        self.gate = nn.Sequential(nn.Linear(gate_in,hid), nn.GELU(), nn.Linear(hid,1), nn.Sigmoid())

    @staticmethod
    def _to_time(x): return x.permute(0,2,1).contiguous()
    @staticmethod
    def _to_ch(x):   return x.permute(0,2,1).contiguous()

    def _band_stats(self, cds):
        feats=[]
        for x in cds:
            feats.append(torch.stack([band_energy(x), band_sfm(x)], dim=-1))
        return torch.stack(feats, dim=1)

    def _per_band_attn(self, cds):
        outs=[]
        for x in cds:
            outs.append(self._to_ch(self.tiny(self._to_time(x))))
        return outs

    def _gate_weights(self, stats):
        B,J,_ = stats.size()
        I = torch.eye(J, device=stats.device).unsqueeze(0).expand(B,-1,-1)
        feats = torch.cat([stats, I], dim=-1)
        raw = self.gate(feats).squeeze(-1)
        return torch.softmax(raw, dim=-1)

    def _apply_scales(self, cds, gates):
        outs=[]; B=cds[0].size(0); C=cds[0].size(1)
        for i,x in enumerate(cds):
            g = gates[:,i].view(B,1,1)
            s = self.band_scales[i].view(1,C,1)
            outs.append(x*g*s)
        return outs

    def forward(self, x, return_regs=False):
        B,T,C = x.shape
        x_ext = self.bd(x)
        X = self._to_ch(x_ext)
        ca, cds = self.fwd(X)
        cds = self._per_band_attn(cds)
        stats = self._band_stats(cds)
        gates = self._gate_weights(stats)
        cds2 = self._apply_scales(cds, gates)
        recon = self.inv((ca, cds2))
        y_ext = self._to_time(recon)
        if self.not_restore:
            y = y_ext
        else:
            y = y_ext + (self.sqrt_beta**2) * (x_ext - y_ext)
        y = self.bd.crop(y, T)
        y = self.do(y)
        y = self.ln(y + x)
        aux=None
        if self.gate_l1>0.0 or self.gate_entropy>0.0 or return_regs:
            l1 = gates.abs().mean()
            ent = -(gates * (gates+1e-8).log()).sum(dim=-1).mean()
            reg = self.gate_l1*l1 + self.gate_entropy*ent
            aux={'gates':gates,'l1_gates':l1,'entropy_gates':ent,'reg_sum':reg}
        return (y, aux) if return_regs else y

class SAARecLayer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        self.saa = SAASubbandsMixer(
            d=d, J=cfg.get('decomp_level',3), wave=cfg.get('wave','db4'),
            p=cfg.get('dropout',0.2), L=cfg.get('boundary_token_len',4),
            reflect=cfg.get('use_reflect_pad', False),
            not_restore=cfg.get('not_restore', False),
            gate_l1=cfg.get('gate_l1', 0.0), gate_entropy=cfg.get('gate_entropy', 0.0)
        )
        self.ffn = FFN(d, p=cfg.get('dropout',0.2))
    def forward(self, x, return_regs=False):
        out = self.saa(x, return_regs=return_regs)
        if isinstance(out, tuple):
            y, aux = out
            y = self.ffn(y); 
            return y, aux
        y = self.ffn(out)
        return y

class SAARec(nn.Module):
    def __init__(self, num_items, hidden_size=128, num_layers=3, max_seq_length=100, dropout=0.2, **saa_cfg):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_seq_length
        self.d = hidden_size
        self.item_emb = nn.Embedding(num_items+1, hidden_size, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_length, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([SAARecLayer(hidden_size, dict(saa_cfg, dropout=dropout)) for _ in range(num_layers)])
        self.ln = TFLN(hidden_size)
        self.out_bias = nn.Parameter(torch.zeros(num_items+1))
    def encode(self, seq_ids, attn_mask):
        B,T = seq_ids.size()
        pos = torch.arange(T, device=seq_ids.device).unsqueeze(0).expand(B,-1)
        x = self.item_emb(seq_ids) + self.pos_emb(pos)
        x = self.drop(x)
        aux_sum = None
        for layer in self.layers:
            out = layer(x, return_regs=True)
            if isinstance(out, tuple):
                x, aux = out
                if aux is not None and 'reg_sum' in aux:
                    aux_sum = (aux_sum or 0.0) + aux['reg_sum']
            else:
                x = out
        x = self.ln(x)
        return x, aux_sum
    def forward(self, seq_ids, attn_mask):
        enc, aux = self.encode(seq_ids, attn_mask)
        lengths = attn_mask.long().sum(dim=1)
        idx = (lengths-1).clamp(min=0).view(-1,1,1).expand(-1,1,self.d)
        last = torch.gather(enc, 1, idx).squeeze(1)
        logits = last @ self.item_emb.weight.t() + self.out_bias
        return logits, aux

@register('model', 'SAARec')
def build_saarec(**cfg):
    return SAARec(**cfg)

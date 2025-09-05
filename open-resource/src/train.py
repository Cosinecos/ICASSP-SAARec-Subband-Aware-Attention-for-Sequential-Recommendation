import torch, time
from .utils.schedule import CosineWithWarmup
from .utils.metrics import evaluate_ranking
from .utils.checkpoint import save_ckpt
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, device, max_grad_norm=5.0):
    model.train()
    ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    total_loss=0.0; n=0
    for batch in tqdm(loader, desc='train', leave=False):
        seq = batch['seq'].to(device)
        mask= batch['mask'].to(device)
        tgt = batch['target'].to(device)
        logits, aux = model(seq, mask)
        loss = ce(logits, tgt)
        if aux is not None:
            loss = loss + aux
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()*seq.size(0); n+=seq.size(0)
    return total_loss/max(1,n)

@torch.no_grad()
def evaluate(model, loader, device, topk=(10,20)):
    model.eval()
    import numpy as np
    HR = {K:[] for K in topk}; NDCG = {K:[] for K in topk}
    for batch in tqdm(loader, desc='eval', leave=False):
        seq = batch['seq'].to(device)
        mask= batch['mask'].to(device)
        tgt = batch['target'].cpu().numpy()
        logits, _ = model(seq, mask)
        scores = logits.cpu().numpy()
        for i in range(len(tgt)):
            res = evaluate_ranking(scores[i], int(tgt[i]), topk=topk)
            for K,(hr,nd) in res.items():
                HR[K].append(hr); NDCG[K].append(nd)
    out = {f'HR@{K}': float(np.mean(HR[K])) for K in topk}
    out.update({f'NDCG@{K}': float(np.mean(NDCG[K])) for K in topk})
    return out

def fit(model, train_loader, val_loader, test_loader, device, cfg, logger):
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    sch = CosineWithWarmup(opt, cfg['train']['epochs'], cfg['train']['warmup_epochs'], cfg['train']['lr'])
    best_metric = -1.0; best_state=None; no_improve=0
    for ep in range(1, cfg['train']['epochs']+1):
        t0=time.time(); tr_loss = train_one_epoch(model, train_loader, opt, device, cfg['train']['max_grad_norm'])
        lr = sch.step(); val_met = evaluate(model, val_loader, device, tuple(cfg['eval']['topk']))
        score = val_met.get('NDCG@10', 0.0) + val_met.get('NDCG@20', 0.0)
        dt=time.time()-t0; logger.write(f"[Epoch {ep:02d}] loss={tr_loss:.4f} lr={lr:.5f} val={val_met} time={dt:.1f}s\n")
        if score > best_metric:
            best_metric = score; best_state = { 'ep':ep, 'model': model.state_dict() }; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg['train']['early_stop_patience']:
                logger.write('Early stop triggered.\n'); break
    if best_state is not None:
        model.load_state_dict(best_state['model'])
    test_met = evaluate(model, test_loader, device, tuple(cfg['eval']['topk']))
    logger.write(f"[BEST @Ep{best_state['ep'] if best_state else '-'}] test={test_met}\n")
    save_ckpt({'cfg':cfg, 'state_dict': model.state_dict()}, 'outputs/best.ckpt')
    return test_met

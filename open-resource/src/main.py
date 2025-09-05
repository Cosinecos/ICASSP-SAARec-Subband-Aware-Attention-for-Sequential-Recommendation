import os, yaml, argparse, torch, importlib
from torch.utils.data import DataLoader
from .utils.seed import set_seed
from .utils.logging import Logger
from .utils.registry import build
from .train import fit

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='path to YAML config')
    return ap.parse_args()

def auto_device(dev_pref):
    if dev_pref=='auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return dev_pref

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    set_seed(cfg.get('seed',42))
    device = auto_device(cfg.get('device','auto'))
    os.makedirs('outputs', exist_ok=True)
    logger = Logger('outputs'); logger.write(f'Using device: {device}\n')

    # import modules to register datasets/models
    importlib.import_module('src.data.datasets')
    importlib.import_module('src.models.saa_rec')

    data_cfg = cfg['data']
    ds_builder = build('dataset', data_cfg['name'])
    train_set, val_set, test_set, meta = ds_builder(data_cfg['data_dir'], data_cfg['max_len'])
    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    model_cfg = cfg['model'].copy(); model_cfg['num_items'] = meta.get('num_items', model_cfg.get('num_items'))
    model = build('model', model_cfg['name'])(**model_cfg).to(device)
    logger.write(f"Model: {model.__class__.__name__} | #params={sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
    test_met = fit(model, train_loader, val_loader, test_loader, device, cfg, logger)
    logger.write('Training done.\n'); logger.close()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse, os, json
from collections import defaultdict

def parse_ratings(path):
    users = defaultdict(list)
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            uid, mid, rating, ts = parts
            try:
                ts = int(ts)
            except:
                continue
            users[uid].append((ts, mid))
    return users

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True, help='folder containing ratings.dat')
    ap.add_argument('--out_dir', required=True, help='output folder')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    users = parse_ratings(os.path.join(args.in_dir, 'ratings.dat'))
    item_map = {}
    def iid(x):
        if x not in item_map:
            item_map[x] = len(item_map)+1
        return item_map[x]

    tr = open(os.path.join(args.out_dir, 'train.jsonl'), 'w', encoding='utf-8')
    va = open(os.path.join(args.out_dir, 'val.jsonl'), 'w', encoding='utf-8')
    te = open(os.path.join(args.out_dir, 'test.jsonl'), 'w', encoding='utf-8')
    ntr=nva=nte=0
    for u, ev in users.items():
        ev.sort()
        items = [iid(m) for _, m in ev]
        if len(items) < 3: 
            continue
        # train (prefix-leave-one)
        for i in range(1, len(items)-1):
            rec = {"user": u, "seq": items[:i], "target": items[i]}
            tr.write(json.dumps(rec, ensure_ascii=False)+'\n'); ntr+=1
        # val/test (true leave-one-out)
        va.write(json.dumps({"user": u, "seq": items[:-2], "target": items[-2]}, ensure_ascii=False)+'\n'); nva+=1
        te.write(json.dumps({"user": u, "seq": items[:-1], "target": items[-1]}, ensure_ascii=False)+'\n'); nte+=1
    tr.close(); va.close(); te.close()

    meta = {"num_users": len(users), "num_items": len(item_map)}
    with open(os.path.join(args.out_dir, 'meta.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    print(f"Done. train={ntr}, val={nva}, test={nte}, num_items={len(item_map)}")

if __name__ == '__main__':
    main()

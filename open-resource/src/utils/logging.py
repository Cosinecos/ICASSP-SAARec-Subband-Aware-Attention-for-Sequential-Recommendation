from datetime import datetime
import os, sys
class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.path = os.path.join(log_dir, f'run-{ts}.log')
        self.f = open(self.path, 'w', encoding='utf-8')
    def write(self, s: str):
        sys.stdout.write(s); sys.stdout.flush()
        self.f.write(s); self.f.flush()
    def close(self): self.f.close()

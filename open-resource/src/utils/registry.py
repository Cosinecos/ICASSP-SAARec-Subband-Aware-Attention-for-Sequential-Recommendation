_REG = {'model': {}, 'dataset': {}}
def register(kind, name):
    def deco(fn):
        _REG[kind][name] = fn
        return fn
    return deco
def build(kind, name):
    if name not in _REG[kind]:
        raise KeyError(f"{kind} '{name}' not registered; available: {list(_REG[kind].keys())}")
    return _REG[kind][name]

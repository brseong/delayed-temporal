from typing import Callable

import torch
from utils.model import L2Net, AbstractL2Net, AbstractExpSubNet
from json import load

def load_l2net_model(hex:str, device=torch.device("cuda")) -> tuple[L2Net, dict[str, object]]:
    path = f"models/{hex}"
    l2net_cfg = torch.load(f"{path}/model.cfg")
    assert load_contents()[hex]["model"] == "L2Net", "The loaded model configuration is not for L2Net."
    
    l2net = L2Net(l2net_cfg["time_steps"],
                l2net_cfg["vector_dim"],
                l2net_cfg["time_steps"]-1,
                l2net_cfg["jeffress_compression"],
                accelerated=True)
    l2net = l2net.to(device).eval()
    l2net.load_state_dict(torch.load(f"{path}/model.pt", map_location=device), strict=True)

    print("L2Net loaded with following configuration:")
    print(l2net_cfg)
    return l2net, l2net_cfg

def load_abst_l2net_model(hex:str, device=torch.device("cuda")) -> tuple[AbstractL2Net, dict[str, object]]:
    path = f"models/{hex}"
    cfg = torch.load(f"{path}/model.cfg")
    assert load_contents()[hex]["model"] == "AbstractL2Net", "The loaded model configuration is not for AbstractL2Net."
    
    l2net = AbstractL2Net(cfg["time_steps"],
                cfg["time_steps"]-1,
                cfg["jeffress_compression"])
    l2net = l2net.to(device).eval()
    l2net.load_state_dict(torch.load(f"{path}/model.pt", map_location=device), strict=True)
    
    print("AbstractL2Net loaded with following configuration:")
    print(cfg)
    return l2net, cfg

def load_abst_expsub_model(hex:str, device=torch.device("cuda")) -> tuple[AbstractExpSubNet, dict[str, object]]:
    path = f"models/{hex}"
    cfg = torch.load(f"{path}/model.cfg")
    assert load_contents()[hex]["model"] == "AbstractExpSubNet", "The loaded model configuration is not for AbstractExpSubNet."
    l2net = AbstractExpSubNet(cfg["time_steps"],
                max_diff=cfg["max_diff"],
                jeffress_compression=cfg["jeffress_compression"])
    l2net = l2net.to(device).eval()
    l2net.load_state_dict(torch.load(f"{path}/model.pt", map_location=device), strict=True)
    
    print("AbstractExpSubNet loaded with following configuration:")
    print(cfg)
    return l2net, cfg

def load_contents():
    with open("models/contents.json", "r") as f:
        contents = load(f)
    return contents

class NetCache[T: AbstractL2Net | AbstractExpSubNet]:
    def __init__(self, model_load_fn: Callable[[str, torch.device], tuple[T, dict[str, object]]]=load_abst_l2net_model):
        self.cache = {}
        self._hex = None
        self._parallel = True
        self._model_load_fn = model_load_fn

    @property
    def hex(self):
        assert self._hex is not None, "Hash is not registered."
        return self._hex
    @hex.setter
    def hex(self, hex:str):
        assert self._hex is None, "Hash is already registered."
        self._hex = hex
        
    @property
    def parallel(self):
        return self._parallel
    @parallel.setter
    def parallel(self, parallel:bool):
        self._parallel = parallel
        
    def __getitem__(self, device:torch.device) -> tuple[T, dict[str, object]]:
        if device not in self.cache:
            self.cache[device] = self._model_load_fn(self.hex, device)
        return self.cache[device]

if __name__ == "__main__":
    contents = load_contents()
    model, cfg = load_abst_expsub_model("ab658c34213811f197a80242ac11000e", torch.device("cpu"))
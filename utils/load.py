import torch
from utils.model import L2Net
from json import load

def load_l2net_model(hex:str, parallel:bool=True) -> tuple[dict[torch.device, L2Net], dict[str, object]]:
    path = f"models/{hex}"
    l2net_cfg = torch.load(f"{path}/model.cfg")
    l2nets = {}
    
    dev_counts = torch.cuda.device_count() if parallel else 1
    for device in range(dev_counts):
        l2net = L2Net(l2net_cfg["TIME_STEPS"],
                    l2net_cfg["VECTOR_DIM"],
                    l2net_cfg["TIME_STEPS"]-1,
                    l2net_cfg["JEFFRESS_COMPRESSION"],
                    accelerated=True).eval()
        l2net.load_state_dict(torch.load(f"{path}/model.pt", map_location=torch.device(f"cuda:{device}")), strict=True)
        l2nets[torch.device(f"cuda:{device}")] = l2net.to(torch.device(f"cuda:{device}"))
    print("L2Net loaded with following configuration:")
    print(l2net_cfg)
    return l2nets, l2net_cfg

def load_contents():
    with open("models/contents.json", "r") as f:
        contents = load(f)
    return contents
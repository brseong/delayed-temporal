import torch
from utils.model import L2Net, AbstractL2Net
from json import load

def load_l2net_model(hex:str, device=torch.device("cuda")) -> tuple[L2Net, dict[str, object]]:
    path = f"models/{hex}"
    l2net_cfg = torch.load(f"{path}/model.cfg")
    assert load_contents()[hex]["model"] == "L2Net", "The loaded model configuration is not for L2Net."
    
    l2net = L2Net(l2net_cfg["TIME_STEPS"],
                l2net_cfg["VECTOR_DIM"],
                l2net_cfg["TIME_STEPS"]-1,
                l2net_cfg["JEFFRESS_COMPRESSION"],
                accelerated=True)
    l2net = l2net.to(device).eval()
    l2net.load_state_dict(torch.load(f"{path}/model.pt", map_location=device), strict=True)

    print("L2Net loaded with following configuration:")
    print(l2net_cfg)
    return l2net, l2net_cfg

def load_abst_l2net_model(hex:str, device=torch.device("cuda")) -> tuple[AbstractL2Net, dict[str, object]]:
    path = f"models/{hex}"
    l2net_cfg = torch.load(f"{path}/model.cfg")
    assert load_contents()[hex]["model"] == "AbstractL2Net", "The loaded model configuration is not for AbstractL2Net."
    
    l2net = AbstractL2Net(l2net_cfg["TIME_STEPS"],
                l2net_cfg["TIME_STEPS"]-1,
                l2net_cfg["JEFFRESS_COMPRESSION"])
    l2net = l2net.to(device).eval()
    l2net.load_state_dict(torch.load(f"{path}/model.pt", map_location=device), strict=True)
    
    print("AbstractL2Net loaded with following configuration:")
    print(l2net_cfg)
    return l2net, l2net_cfg


def load_contents():
    with open("models/contents.json", "r") as f:
        contents = load(f)
    return contents
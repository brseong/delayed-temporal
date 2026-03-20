# %%
import argparse
import torch, wandb
import numpy as np
import matplotlib.pyplot as plt
from math import log
from jaxtyping import Float
from tqdm.auto import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from spikingjelly.activation_based.neuron import BaseNode
from utils.model import AbstractExpSubNet, L2Net, AbstractL2Net
from utils.load import load_abst_expsub_model, load_l2net_model, load_abst_l2net_model
from utils.datasets import generate_expsub_dataset, generate_l2_square_dataset, encode_temporal_np
# %%
# torch.autograd.set_detect_anomaly(True)
writer = SummaryWriter(log_dir="./runs/l2_distance_experiment")

np.random.seed(42)
rng = torch.manual_seed(42)

# %%
torch.set_printoptions(threshold=10_000, precision=2, linewidth=160, sci_mode=False)

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="L2 distance training script")
    parser.add_argument("--num-samples", type=int, default=100000, help="Number of samples to generate for training.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-1, help="Learning rate.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model after training.")
    parser.add_argument("--hex", type=str, default="", help="Hex identifier for the model.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the training on.")
    parser.add_argument("--initial-alpha", type=float, default=2.0, help="Initial value for the alpha parameter.")
    parser.add_argument("--max-alpha", type=float, default=10.0, help="Maximum value for the alpha parameter.")
    parser.add_argument("--time-steps", type=int, default=20, help="Number of time steps.")
    parser.add_argument("--jeffress-compression", type=int, default=1, help="Jeffress compression factor.")
    
    
    subparsers = parser.add_subparsers(dest="dataset", required=True, help="Jeffress model type: l2_square or expsub")
    
    # Subparser for l2_square dataset
    l2_square_parser = subparsers.add_parser("l2_square", help="Train with L2 square dataset")
    l2_square_parser.add_argument("--vector-dim", type=int, default=64, help="Dimensionality of the input vectors.")
    l2_square_parser.add_argument("--min-val", type=float, default=-8.0, help="Minimum value for the input vectors.")
    l2_square_parser.add_argument("--max-val", type=float, default=8.0, help="Maximum value for the input vectors.")
    
    expsub_parser = subparsers.add_parser("expsub", help="Train with ExpSub dataset")
    expsub_parser.add_argument("--max-diff", type=float, default=30 + log(197), help="Maximum difference for the input vectors.")

    return parser.parse_args()

cfg = vars(parse_args())

# %%
NUM_SAMPLES = int(cfg["num_samples"])
DATASET = cfg["dataset"]
NUM_EPOCHS = int(cfg["num_epochs"])
TIME_STEPS = int(cfg["time_steps"])
JEFFRESS_COMPRESSION = int(cfg["jeffress_compression"])
BATCH_SIZE = int(cfg["batch_size"])
INITIAL_ALPHA = float(cfg["initial_alpha"])
MAX_ALPHA = float(cfg["max_alpha"])
LR = float(cfg["lr"])
EVAL = bool(cfg["eval"])
device = cfg["device"]

match DATASET:
    case "l2_square":
        VECTOR_DIM = int(cfg["vector_dim"])
        MIN_VAL = float(cfg["min_val"])
        MAX_VAL = float(cfg["max_val"])
        model = AbstractL2Net(TIME_STEPS,
                    jeffress_radius=TIME_STEPS-1,
                    jeffress_compression=JEFFRESS_COMPRESSION).to(device)
        X_data, y_data = generate_l2_square_dataset(NUM_SAMPLES, VECTOR_DIM, low=MIN_VAL, high=MAX_VAL, normalize=False)  # X_data: N 2 D, y_data: N
        X_data = (torch.FloatTensor(X_data) - MIN_VAL) / (MAX_VAL - MIN_VAL)  # N 2 D
        y_data = torch.FloatTensor(y_data) / (2 * (VECTOR_DIM**.5)) # N D
        ### For non-abstract L2Net model, encode the input data into temporal spike patterns using latency coding (TTFS).
        # X_data_temporal = torch.stack([torch.FloatTensor(encode_temporal_np(X_data[:,0,:], TIME_STEPS, 0, min_val=MIN_VAL, max_val=MAX_VAL)),
        #                       torch.FloatTensor(encode_temporal_np(X_data[:,1,:], TIME_STEPS, 0, min_val=MIN_VAL, max_val=MAX_VAL))],
        # dataset = torch.utils.data.TensorDataset(X_data_temporal.transpose(1, 0), y_data, torch.tensor(X_data))  # T N 2D -> N T 2D
    case "expsub":
        MAX_DIFF = float(cfg["max_diff"])
        model = AbstractExpSubNet(TIME_STEPS,
                    max_diff=MAX_DIFF, # TimeBounds(min=0, max=35.28320372873799)
                    jeffress_compression=JEFFRESS_COMPRESSION).to(device)
        X_data, y_data = generate_expsub_dataset(NUM_SAMPLES, minus_logmin=30.0, seqlen=197)
        X_data = torch.FloatTensor(X_data) # N 2 D
        y_data = torch.FloatTensor(y_data) # N
    case _:
        raise ValueError(f"Unsupported dataset type: {DATASET}")
print(model)

dataset = torch.utils.data.TensorDataset(X_data, y_data, torch.tensor(X_data))  # N 2D
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
    )
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
current_alpha = INITIAL_ALPHA
loss = torch.tensor(float("inf"))

pred_hist, target_hist, err_hist = [], [], []
train_step = 0
eval_step = 0
substep = 0
eval_substep = 0
if not EVAL:
    with wandb.init(project=f"DelayedTemporal_{DATASET}",
                    config=cfg) as run:
        run.define_metric("train/*", step_metric="train_step")
        run.define_metric("delay/*", step_metric="train_step")
        run.define_metric("SDC/*", step_metric="train_step")
        run.define_metric("Neuron/*", step_metric="substep")
        run.define_metric("eval/*", step_metric="eval_step")
        run.define_metric("eval_sub/*", step_metric="eval_substep")
        for epoch in tqdm(range(NUM_EPOCHS)):
            model.train()
            
            # current_alpha = INITIAL_ALPHA + (MAX_ALPHA - INITIAL_ALPHA) * epoch / NUM_EPOCHS
            # for m in model.modules():
            #     if isinstance(m, BaseNode):
            #         if hasattr(m.surrogate_function, "alpha"):
            #             setattr(m.surrogate_function, "alpha", min(current_alpha, 10.0))
                        
            for i, batch in enumerate(pbar:=tqdm(train_loader, leave=False)):
                # inputs:Float[Tensor, "N T 2 D"]; targets:Float[Tensor, "N D"]
                inputs:Float[Tensor, "N 2 D"]; targets:Float[Tensor, "N D"]
                inputs, targets, input_raw = batch
                inputs = inputs.to(device); targets = targets.to(device); input_raw = input_raw.to(device)
                # out = model(inputs.transpose(1, 0)) # N T 2 D -> T N 2 D -> model -> T N 1
                out = model(inputs) # N 2 D -> model -> N 1
                pred = out
                loss = criterion(pred, targets.reshape(pred.shape))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({"loss": loss.item(), "scale":(out.max()-out.min()).item(), "lr": scheduler.get_last_lr()[0]})
                if i % 10 == 0:
                    run.log({
                        "train_step": (train_step := train_step + 1),
                        "train/loss": loss.item(),
                        "train/err":(pred - targets).abs().mean().item(),
                        "train/alpha": current_alpha,
                    # }|{
                    #         f"SDC/rate_{i}": torch.stack(model.stats['jeffress_model.2.neuron'], dim=0).mean(dim=(0,1,2))[i] for i in range(2*TIME_STEPS - 1)
                            }
                    )
                    # for t in range(2*TIME_STEPS):
                    #     run.log({
                    #         "substep": (substep := substep + 1),
                    #     }
                    #             |{
                    #         f"Neuron/I_{j}": model.jeffress_model[2].i_seq[t][0,0,j] for j in range(2*TIME_STEPS - 1)
                    #         }
                    #             |{
                    #         f"Neuron/V_{j}": model.jeffress_model[2].v_seq[t][0,0,j] for j in range(2*TIME_STEPS - 1)
                    #         })
            scheduler.step()
            
            with torch.no_grad():
                model.eval()
                for batch in (pbar:=tqdm(test_loader, leave=False)):
                    inputs, targets, input_raw = batch
                    inputs = inputs.to(device); targets = targets.to(device); input_raw = input_raw.to(device)
                    # out = model(inputs.transpose(1, 0)) # NT(2D)->TN(2D)->model->N  
                    out = model(inputs) # N 2 D -> model -> N 1
                    pred = out
                    loss = criterion(pred, targets.reshape(pred.shape))
                    pred_hist.extend(pred.squeeze().tolist())
                    target_hist.extend(targets.squeeze().tolist())
                    err_hist.extend(torch.abs(pred.squeeze() - targets.squeeze()).tolist())
                    pbar.set_postfix({"loss": loss.item(), "pred": pred_hist[-1], "target": target_hist[-1]})
                    
                    run.log({"eval_step": (eval_step := eval_step + 1),
                            "eval/loss": loss.item()})
                    for n in range(pred.shape[0]):
                        run.log({"eval_substep": (eval_substep := eval_substep + 1),
                            "eval_sub/err":  (pred - targets.reshape(pred.shape)).abs()[n].item()})
else:
    hex = cfg["hex"]
    match DATASET:
        case "l2_square":    
            model, cfg = load_abst_l2net_model(hex, torch.device("cuda"))
        case "expsub":
            model, cfg = load_abst_expsub_model(hex, torch.device("cuda"))
        case _:
            raise ValueError(f"Unsupported dataset type: {DATASET}")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, targets, input_raw = batch
            inputs = inputs.to(device); targets = targets.to(device); input_raw = input_raw.to(device)
            # out = model(inputs.transpose(1, 0)) # NT(2D)->TN(2D)->model->N
            out = model(inputs) # N 2 D -> model -> N 1
            pred = out
            loss = criterion(pred, targets)
            pred_hist.extend(pred.view(-1).tolist())
            target_hist.extend(targets.view(-1).tolist())
            err_hist.extend(torch.abs(pred.view(-1) - targets.view(-1)).tolist())
    print(f"Test MSE Loss: {loss.item():.6f}, MAE: {np.mean(err_hist):.6f}")

# %%
if not EVAL:
    from uuid import uuid1
    from json import load, dump
    from pathlib import Path

    save_id = uuid1().hex
    save_dir = Path(f"models/{save_id}")

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    with open("models/contents.json", "r", encoding="utf-8") as f:
        contents = load(f)

    contents |= ({save_id: {
        "model": model.__class__.__name__,
        "final_loss": loss.item(),
        "final_mae": np.mean(err_hist)} | cfg})

    with open("models/contents.json", "w", encoding="utf-8") as f:
        dump(contents, f, indent=4)

    torch.save(model.state_dict(), save_dir / "model.pt")
    torch.save(cfg, save_dir / "model.cfg")
    print(f"Model saved to {save_dir}/model.pt")

# %%
plt.title("MAE (Mean Absolute Error)")
plt.plot(err_hist, linewidth=0.025, label="Difference")
# plt.ylim(-.1, 1)
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.legend()

# %%
# y_data

# %%
plt.hist(err_hist[100000:])
plt.savefig("error_histogram.png")
# %%




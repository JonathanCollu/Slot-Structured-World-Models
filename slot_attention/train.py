import argparse
from slot_attention.slot_attention import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from os import makedirs
from os.path import exists
import wandb
import json
from torch.utils import data
from utils import StateTransitionsDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

parser = argparse.ArgumentParser()

parser.add_argument('--config', default=None, type=str, help='name of the configuration to use' )
parser.add_argument('--init_ckpt', default=None, type=str, help='initial weights to start training')
parser.add_argument('--ckpt_path', default='checkpoints/spriteworld/', type=str, help='where to save models' )
parser.add_argument('--ckpt_name', default='model', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_path', default='spriteworld/spriteworld/data', type=str, help='Path to the data' )
parser.add_argument('--resolution', default=[35, 35], type=list)
parser.add_argument('--data_size', default=6400, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use. Choose between [adam, sgd, rmsprop, adamw, radam]' )
parser.add_argument('--small_arch', action='store_true', help='if true set the encoder/decoder dim to 32, 64 otherwise')
parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--wandb_project', default=None, type=str, help='wandb project')
parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity')

args = parser.parse_args()
args = vars(args)


if args["config"] is not None:
    args["ckpt_name"] = args["config"]
    with open("configs.json", "r") as config_file:
        configs = json.load(config_file)[args["config"]]
    for key, value in configs.items():
        try:
            args[key] = value
        except KeyError:
            exit(f"{key} is not a valid parameter")

use_wandb = args["wandb_project"] is not None and args["wandb_entity"] is not None

if not exists(args["ckpt_path"]):
    makedirs(args["ckpt_path"])

model = SlotAttentionAutoEncoder(
            tuple(args["resolution"]), args["num_slots"], args["num_iterations"], 
            args["slots_dim"], 32 if args["small_arch"] else 64, args["small_arch"]
        ).to(device)
model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)  # model.to(device) do not move
model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)  # these tensors automatically

if args["init_ckpt"] is not None:
    checkpoint = torch.load(args["ckpt_path"]+args["init_ckpt"]+".ckpt")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

criterion = torch.nn.MSELoss()

params = [{'params': model.parameters()}]

if use_wandb:
    wandb.init(project=args["wandb_project"], entity=args["wandb_entity"])
    wandb.run.name = args["config"]
    logs = {}
    for key, value in args.items():
        logs[key] = value
    wandb.config = logs
    wandb.watch(model)

dataset = StateTransitionsDataset(hdf5_file=args["data_path"])
train_dataloader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0)

if args["config"] is not None:
    if args["optimizer"] == "adam":
        optimizer = optim.Adam(params, lr=args["learning_rate"])
    elif args["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(params, lr=args["learning_rate"])
    elif args["optimizer"] == "sgd":
        optimizer = optim.SGD(params, lr=args["learning_rate"])
    else:
        exit("Select a valid optimizer.")
else:
    optimizer = optim.Adam(params, lr=args["learning_rate"])

if args["init_ckpt"] is not None:
    try: optimizer.load_state_dict(checkpoint["optim_state_dict"])
    except: pass

start = time.time()
if args["init_ckpt"] is not None:
    try:
        epoch, i = checkpoint["epoch"]
        epoch += 1
    except: epoch, i = 0, 0
else:
    epoch, i = 0, 0
steps = args["data_size"] // args["batch_size"]

model.train()
for epoch in range(epoch, args["num_epochs"]):
    total_loss = 0
    for sample in tqdm(train_dataloader, position=0):
        i += 1
        if i < args["warmup_steps"]:
            learning_rate = args["learning_rate"] * (i / args["warmup_steps"])
        else:
            learning_rate = args["learning_rate"]

        learning_rate = learning_rate * (args["decay_rate"] ** (
            i / args["decay_steps"]))

        optimizer.param_groups[0]['lr'] = learning_rate

        image = sample[0].permute((0,3,1,2))
        image = image + (torch.randint(0, 3, (1,)) > 0)*0.5*torch.rand((1, 3, 1, 1)).clip(0, 1)
        image = image.to(device)
        del sample
        
        recon_combined, recons, masks, slots = model(image)
        
        loss = criterion(recon_combined, image)
        
        if use_wandb: wandb.log({"rec_loss": loss}, step=i)
        
        total_loss += loss.item()

        del recon_combined, recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(train_dataloader)

    print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
        datetime.timedelta(seconds=time.time() - start)))

    if not epoch % 1:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": (epoch, i)
            }, args["ckpt_path"]+args["ckpt_name"]+"_"+str(epoch)+"ep.ckpt")
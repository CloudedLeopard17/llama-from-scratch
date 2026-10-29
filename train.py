import os
import torch
import math
import time
import sys
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from model import Transformer, load_model, save_model

from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#defualt is highest
torch.set_float32_matmul_precision('high')


@dataclass
class CFG:
    ddp = True
    #defaults for ddp 
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    data_path : str ='./data/tokenized_fineweb.bin'
    experiment_name: str = "fineweb"
    vocab_size: int = 32000
    valid_size: int = 20_000_000
    
    max_seq_len: int = 1024
    dim: int = 768
    n_layers: int = 12
    n_heads_q: int = 12
    n_heads_kv: Optional[int] = 4
    dropout: float = 0.0  #no dropout as in the original paper
    multiple_of: int = 256
    ffn_multiplier: Optional[int] = None
    norm_eps: float = 1e-6
    init_std: float = 0.02

    state_dir: str = "model_states_tiny_fineweb"
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    batch_size_train: int = 24
    batch_size_valid: int = 32
    valid_batches: Optional[int] = 50
    num_workers: int = 10
    n_epochs: int = 2
    last_state: Optional[str] = None
    best_state_file_name: str = "best_model"
    last_state_file_name: str = "last_model"

    init_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1000
    max_steps: int = 107123 
    valid_every_steps: int = 1000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)

args = CFG()

if args.ddp:
    args.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if args.ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    args.ddp_rank = ddp_rank
    args.ddp_local_rank = ddp_local_rank
    args.ddp_world_size = ddp_world_size
    args.master_process = master_process
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

def lr_at(step: int, cfg: CFG) -> float:
    """Linear warmup for cfg.warmup_steps, then cosine decay to cfg.min_lr."""
    if step < cfg.warmup_steps:
        return cfg.init_lr*(step+1)/cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps)/max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5*(cfg.init_lr - cfg.min_lr)*(1.0 + math.cos(math.pi*progress))

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path: str, block_size: int, args: CFG, train: bool= True):
        self.data = np.memmap(data_path, dtype=np.uint16)[: -args.valid_size] if train\
              else np.memmap(data_path, dtype=np.uint16)[-args.valid_size:] 
        
        self.block_size = block_size
        self.train = train

    def __len__(self):
        return len(self.data)//self.block_size-1

    def __getitem__(self, idx):
        x = self.data[idx*self.block_size:(idx+1)*self.block_size]
        y = self.data[idx*self.block_size+1:(idx+1)*self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train(
    train_config: CFG,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    loss_fn: torch.nn.modules.loss.CrossEntropyLoss,
    vocab_size: int,
    device: str,
    device_type: str,
    summary_writer: Optional[SummaryWriter],
    global_step: int,
    best_loss: float
    ):

    batch_loss = np.array([])
    norms = np.array([])
    model.train()
    batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch_idx}") if not args.ddp else train_loader
    t0 = time.time()
    tokens_processed = 0
    for step,(x, y) in enumerate(batch_iterator):
        pos = torch.arange(x.shape[1], device=device).repeat(x.shape[0], 1)
        inp = x.to(device, non_blocking=True)
        labels = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = model(inp, pos, mask=None)
            loss = loss_fn(out.view(-1, vocab_size),labels.view(-1))

        batch_loss = np.append(batch_loss, [loss.item()])
        tokens_processed += inp.numel()
        loss = loss / train_config.grad_accumulation_steps
        loss.backward()

        if (step+1)%train_config.grad_accumulation_steps == 0:
            # warmup + cosine, applied per optimizer step
            lr = lr_at(global_step, train_config)
            for group in optimizer.param_groups:
                group['lr'] = lr

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            norms = np.append(norms, [norm.item()])
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # step-based validation
            if (global_step % train_config.valid_every_steps == 0) and args.master_process:
                valid_loss = valid(train_config, model, valid_loader, epoch_idx, loss_fn,
                                   vocab_size, device, device_type)
                window = batch_loss[-train_config.valid_every_steps:].mean()
                note = ''
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    note = '   <- best, saving'
                    save_model(model, optimizer, global_step, device=device,
                               state_file_name=train_config.best_state_file_name,
                               state_dir=train_config.state_dir, loss=best_loss)
                print('step {:>6}  lr {:.2e}  train {:.4f}  valid {:.4f}{}'.format(
                    global_step, lr, window, valid_loss, note))
                summary_writer.add_scalar('Loss/Valid', valid_loss, global_step)
                summary_writer.add_scalar('Loss/TrainWindow', window, global_step)
                summary_writer.add_scalar('Learning Rate', lr, global_step)

            if global_step >= train_config.max_steps:
                break
        if not args.ddp:
            batch_iterator.set_postfix({"loss": f"{loss.item()*train_config.grad_accumulation_steps:.4f}"})

    epoch_loss = torch.tensor(batch_loss, device=device).mean()
    epoch_norm = torch.tensor(norms, device=device).mean()
    if args.ddp:
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(epoch_norm, op=dist.ReduceOp.AVG)
    
    dt = time.time() - t0
    tokens_per_sec = (tokens_processed/dt)*args.ddp_world_size
    if args.master_process:
        print('Epoch: {} Train Loss: {:.3f} Average Norm: {:.3f} Tokens/sec: {:.2f} Time: {:.2f}s'.format(
            epoch_idx, epoch_loss, epoch_norm, tokens_per_sec, dt))
    return epoch_loss, global_step, best_loss

def valid(
    args: CFG,
    model: Transformer,
    valid_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    loss_fn: torch.nn.modules.loss.CrossEntropyLoss,
    vocab_size: int,
    device: str,
    device_type: str,
    )-> float:
    max_batches = args.valid_batches
    batch_loss = np.array([])
    model.eval()
    batch_iterator = valid_loader
    with torch.no_grad():
        for step,(x, y) in enumerate(batch_iterator):
            pos = torch.arange(x.shape[1], device=device).repeat(x.shape[0], 1)
            inp = x.to(device)
            labels = y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                out = model(inp, pos, mask=None)
                loss = loss_fn(out.view(-1, vocab_size),labels.view(-1))

            batch_loss = np.append(batch_loss, [loss.item()])
            if max_batches is not None and step+1 >= max_batches:
                break
    epoch_loss = batch_loss.mean()
    model.train()
    return epoch_loss

def main(
    args: CFG,
    model,
    optimizer: torch.optim.Optimizer,
    dataset_train: torch.utils.data.Dataset,
    dataset_valid: torch.utils.data.Dataset,
    device: str
    ):

    summary_writer = SummaryWriter(log_dir=args.experiment_name) if args.master_process else None
    sampler = DistributedSampler(dataset_train, shuffle=True) if args.ddp else None
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size_train,\
                                  num_workers=args.num_workers,
                                  sampler= sampler,
                                  shuffle= not args.ddp
                                )
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size_valid,
                              num_workers=2, pin_memory=True, drop_last=True)
    loss_fn = nn.CrossEntropyLoss()
    vocab_size = args.vocab_size

    model.to(device)
    best_loss = float(np.inf)
    global_step = 0
    if args.last_state is not None:
        state_path = os.path.join(args.state_dir, args.last_state)
        state = torch.load(state_path, map_location='cpu')
        load_model(model, args.last_state, args.state_dir)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        best_loss = state['loss'].item()
        global_step = state['step'].item()
        if args.ddp:
            sampler.set_epoch(global_step//len(train_loader))
        if args.master_process:
            print(f"Loaded model from {args.last_state}")
            print(f"Best Loss: {best_loss:.4f}, Resuming from step {global_step}")
    if args.master_process:
        print('warmup {} steps, cosine {:.1e} -> {:.1e} over {} steps'.format(
            args.warmup_steps, args.init_lr, args.min_lr, args.max_steps))
        print('validating every {} steps on {} batches\n'.format(
            args.valid_every_steps, args.valid_batches if args.valid_batches is not None else str(len(valid_loader))))

    t_begin = time.time()
    for epoch in range(args.n_epochs):
        if args.ddp:
            sampler.set_epoch(epoch)
        train_loss, global_step, best_loss = train(
            args, model, optimizer, train_loader, valid_loader, epoch, loss_fn,
            vocab_size, device, device_type, summary_writer, global_step, best_loss)
        if args.master_process:
            summary_writer.add_scalar('Loss/Train', train_loss, epoch)
        if global_step >= args.max_steps:
            if args.master_process:
                print('Reached max_steps ({}).'.format(args.max_steps))
            break
    if args.master_process:
        save_model(model, optimizer, global_step, device=device,
                state_file_name=args.last_state_file_name,
                state_dir=args.state_dir, loss=best_loss)
        print("Total time: {:.2f}, Best Valid Loss: {:.3f}".format(time.time() - t_begin, best_loss))

    if args.ddp:
        destroy_process_group()

def get_optimizer_params(model, lr, weight_decay=0.0):
    no_decay = ["bias", "norm"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


d_train = Dataset(args.data_path, args.max_seq_len, args,train=True)
d_valid = Dataset(args.data_path, args.max_seq_len, args,train=False)

model = Transformer(args)
model.to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if args.master_process:
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
model = torch.compile(model)
if args.ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if args.ddp else model # always contains the "raw" unwrapped model


optimizer_params = get_optimizer_params(raw_model, args.init_lr, args.weight_decay)
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == "cuda" 

optimizer = torch.optim.AdamW(
    optimizer_params,
    lr = args.init_lr,
    betas=args.betas,
    fused=use_fused
)


main(args, model, optimizer, d_train, d_valid, device)

#python -m torch.distributed.run --nproc_per_node=2   train.py 

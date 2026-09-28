import os
import argparse
from tqdm import tqdm
import time
import numpy as np
import logging

import model
import utils
from utils.visualizor.utils import plot_loss_distribution
from dataset import DATASET_DICT 
from model import MODEL_DICT

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# torch.autograd.set_detect_anomaly(True)

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="config of task to run")
parser.add_argument("--config", type=str,default="config_glno.yaml", help="path to config file")
parser.add_argument("--evaluate", default=False, action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--load_model", type=str, help="path to load a pretrained model from")
parser.add_argument("--distributed", default=False, action="store_true", help="use distributed training")
parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training (automatically set by torchrun)")
parser.add_argument("--dataset_name",type=str,default="data.h5",help="name of dataset to use")
parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
parser.add_argument("--channels", type=int, default=None, help="number of channels in the input data")
parser.add_argument("--modes", type=int, default=None, help="number of modes in the input data")
parser.add_argument("--sigma", type=int, default=None, help="noise level in the input data")
parser.add_argument("--number_worker", type=int, default=4, help="number_worker")
args = parser.parse_args()
args=vars(args) # convert to dictionary

# paths
base_path = os.path.dirname(__file__)
if args['task']=='cortex' or args['task']=='intra' or args['task']=='poissonunstruc':
    dataset_path = os.path.join(base_path, "data", args['task'], 'processed', args['dataset_name'])
else:
    dataset_path = os.path.join(base_path, "data", args['task'])
args['config'] = os.path.join(base_path, "config", args['task'].split("/")[0], args['config'])
args['log_dir']=os.path.join(base_path,"logs")
args=utils.load_config(args) # TODO load config from pretrained model

if args['channels'] is not None:
    args['model']['width']=args['channels']
    args['model']['C_width']=args['channels']
if args['modes'] is not None:
    args['model']['modes']=args['modes']
    args['model']['modes1']=args['modes']
    args['model']['modes2']=args['modes']
    args['model']['glno_poles']=args['modes']
if args['sigma'] is not None:
    args['model']['sigma']=args['sigma']
    args['model']['num_sigma_i']=args['sigma']
    args['model']['num_sigma_j']=args['sigma']
    args['model']['glno_sigma']=args['sigma']

if args['system']['log']:
    saving_path, saving_name = utils.set_up_logger(args["model"]["name"], args["task"], args['log_dir']) #  (TODO: add more details on names)
else:
    saving_path, saving_name = utils.get_dir_path(args["model"]["name"], args["task"], args['log_dir'])
args['saving_path'] = saving_path
args['saving_name'] = saving_name
if args['system']['log']:
    utils.save_config(args, saving_path)

# initialize distributed training
if args["distributed"]:# TODO check distrubuted
    utils.setup_distributed(args)
    args['system']['verbose'] = False if args['rank']!=0 else args['system']['verbose']

# system info logging
if args['system']['verbose']:
    logger = logging.info if args['system']['log'] else print
    logger(args)

# system settings
device = torch.device(f'cuda:{args["local_rank"]}' if torch.cuda.is_available() else 'cpu')
args["model"]["device"]=device
dtype = torch.float32

utils.set_seed(args['seed'])

# === Load datasets
if not args["evaluate"]: 
    train_dataset = DATASET_DICT[args["dataset"]["name"]](mode='train', dataset_path=dataset_path)
    if args["dataset"]["create_val"]:
        train_list,val_list=utils.split_train_val(len(train_dataset),ratio=0.05)
        train_dataset.split_val(train_list)
    if args["distributed"]:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args["world_size"], rank=args["rank"], shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args["dataset"]["train_batch_size"], sampler=train_sampler,
                              shuffle=(train_sampler is None), collate_fn=utils.COLLATE_Path[args['task']] if args['dataset']['collate']=='pad' else None,
                              num_workers=args['number_worker'])
    if args['training']['train_info']:
        train_loader_info = DataLoader(train_dataset, batch_size=1 if args["dataset"]["train_batch_size"] else None, sampler=train_sampler,
                              shuffle=(train_sampler is None), collate_fn=utils.COLLATE_Path[args['task']] if args['dataset']['collate']=='pad' else None,
                              num_workers=args['number_worker'])

test_dataset = DATASET_DICT[args["dataset"]["name"]](mode='test', dataset_path=dataset_path)
if args["distributed"]:
    test_sampler = DistributedSampler(test_dataset, num_replicas=args["world_size"], rank=args["rank"], shuffle=False)
else:
    test_sampler = None
test_loader = DataLoader(test_dataset, batch_size=args["dataset"]["test_batch_size"], sampler=test_sampler, shuffle=False,
                          collate_fn=utils.COLLATE_Path[args['task']] if args['dataset']['collate']=='pad' else None, num_workers=args['number_worker'])
test_loader_info = DataLoader(test_dataset, batch_size=1 if args["dataset"]["test_batch_size"] else None, sampler=test_sampler, shuffle=False,
                          collate_fn=utils.COLLATE_Path[args['task']] if args['dataset']['collate']=='pad' else None, num_workers=args['number_worker'])

valid_dataset = DATASET_DICT[args["dataset"]["name"]](mode='val', dataset_path=dataset_path)
if args["dataset"]["create_val"] and not args["evaluate"]:
    valid_dataset.split_val(val_list)
if args["distributed"]:
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=args["world_size"], rank=args["rank"], shuffle=False)
else:
    valid_sampler = None
valid_loader = DataLoader(valid_dataset, batch_size=args["dataset"]["test_batch_size"], sampler=valid_sampler, shuffle=False,
                          collate_fn=utils.COLLATE_Path[args['task']] if args['dataset']['collate']=='pad' else None, num_workers=args['number_worker'])
if args['training']['train_info']:
        valid_loader_info = DataLoader(valid_dataset, batch_size=1 if args["dataset"]["test_batch_size"] else None, sampler=valid_sampler, shuffle=False,
                          collate_fn=utils.COLLATE_Path[args['task']] if args['dataset']['collate']=='pad' else None, num_workers=args['number_worker'])

if args['system']['verbose']:
    logger("Train dataset size: {}".format(len(train_dataset)))
    logger("Vali dataset size: {}".format(len(valid_dataset)))
    logger("Test dataset size: {}".format(len(test_dataset)))

# === Create the model
if args['model']['name'] not in MODEL_DICT.keys():
    raise NotImplementedError("Model {} not implemented".format(args['model']['name']))
model = MODEL_DICT[args["model"]["name"]](config=args["model"])

# model trainable parameters number
if args['system']['verbose']:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger(f"Total params: {total / 1e6:.2f} M | Trainable: {trainable / 1e6:.2f} M")

model = model.to(device)

# data distribution initialization
if args["distributed"]:
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

if args["load_model"]: # TODO check distributed model loading and model loading from checkpoint
    if args['system']['verbose']:
        logger("Loading pretrained model from: " + str(args.load_model))
    if args["distributed"]:
        if args["rank"] == 0:
            checkpoint = torch.load(args.load_model)
        else:
            checkpoint = None
       
        checkpoint_list = [checkpoint]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint = checkpoint_list[0]  
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(args.load_model))
    if args['system']['verbose']:
        logger("...done")

if args["evaluate"] and not args["load_model"]:
    raise ValueError("Called with --evaluate but not --load_model. This will evaluate on a randomly initialized model, which is probably not what you want to do.")

# training settings
n_epoch = args["training"]["epoch"]
lr = args["training"]["lr"]
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = utils.build_scheduler(optimizer, args['training']['scheduler'])
if args['system']['monitor']:
    monitor=utils.ComplexGradientMonitor(model,mode='real_imag',log_fn=logger if args['system']['log'] else print,log_interval=1)

if args["training"]["loss"]=="L2relative":
    myloss = utils.LpLoss(size_average=False)
elif args["training"]["loss"]=="cross_entropy":
    myloss = utils.classify_nll_loss()
elif args["training"]["loss"]=="MultiL2relative":
    myloss = utils.MultipleLoss(size_average=False)
elif args['training']['loss']=='CarLoss':
    myloss = utils.CarCFDLoss(size_average=False)
elif args['training']['loss']=='TurbulentLoss':
    myloss = utils.TurbulentLoss(size_average=False)
elif args['training']['loss']=='CFlowLoss':
    myloss = utils.CFlowLoss(size_average=False)
elif args['training']['loss']=='cross_entropy_global':
    myloss= utils.classify_nll_loss_global()
else:
    raise NotImplementedError("Loss {} not implemented".format(args["training"]["loss"]))

# Do a training epoch
def train_process(epoch):
    model.train()
    optimizer.zero_grad()

    if args["distributed"]:
        train_loader.sampler.set_epoch(epoch)
    

    for data in tqdm(train_loader):# if not args['system']['log'] else train_loader:
        preds = model(data)

        # Evaluate loss
        labels=data['labels'].to(device)
        loss_result =myloss(preds,labels,mask=data.get('mask', None))
        
        if isinstance(loss_result, (list, tuple)): #multiple losses
            loss_result[args['training']['train_index']].backward()
            loss_values = [loss_item.item() if hasattr(loss_item, 'item') else loss_item for loss_item in loss_result]
        else: #simgle loss               
            loss_result.backward()                                                
            loss_values = [loss_result.item()] 
        
        if 'loss_history' not in locals():
            loss_history = [[] for _ in range(len(loss_values))]
        
        for i, loss_val in enumerate(loss_values):
            # if len(data['labels'].shape)==1:
            #     num_sample=1
            # else:
            #     num_sample=data['labels'].shape[0]
            loss_history[i].append(loss_val) #*data['labels'].shape[0]
        

        # Step the optimizer
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
        # for p in model.parameters():
        #     if p.grad is not None:
        #         if p.grad.data.isnan().any():
        #             print(loss_result[0].item())
        #             print("Warning: NaN detected in gradient of parameter {}".format(p.name))
        #             p.grad.data.zero_()

        # def compute_max_grad_norm(model):
        #     total_norm = 0.0
        #     max_norm = 0.0
        #     for p in model.parameters():
        #         if p.grad is not None:
        #             param_norm = p.grad.data.norm(2).item()
        #             total_norm += param_norm ** 2
        #             if param_norm > max_norm:
        #                 max_norm = param_norm
        #     total_norm = total_norm ** 0.5
        #     return total_norm, max_norm
        # total_norm, max_norm = compute_max_grad_norm(model)
        # print(f"Total gradient norm: {total_norm:.4f}, Max gradient norm: {max_norm:.4f}")
        
        # 如果梯度爆炸，可以打印更多信息
        # if total_norm > 1e6 or torch.isnan(torch.tensor(total_norm)):
        #     print("警告：梯度爆炸或出现 NaN！")
        #     # 可选：打印每个层的梯度范数
        #     for name, p in model.named_parameters():
        #         if p.grad is not None:
        #             print(f"{name}: grad norm = {p.grad.norm(2).item():.4f}")
        #     # 可以在这里设置断点或直接退出调试
        #     optimizer.zero_grad()
        #     continue

        optimizer.step()
        optimizer.zero_grad()
        
    if args['system']['monitor']:
        monitor.log_step()
        for i in range(len(model.blocks)):
            print(f"{i} mean {torch.mean(model.blocks[i].LT.system_poles.real).item():.5e} min {torch.min(model.blocks[i].LT.system_poles.real).item():.5e} max {torch.max(model.blocks[i].LT.system_poles.real).item():.5e}") #
            print(f"{i} mean {torch.mean(model.blocks[i].LT.system_poles.imag).item():.5e} min {torch.min(model.blocks[i].LT.system_poles.imag).item():.5e} max {torch.max(model.blocks[i].LT.system_poles.imag).item():.5e}") #
            print(f"{i} mean {torch.mean(model.blocks[i].LT.sigma).item():.5e} min {torch.min(model.blocks[i].LT.sigma).item():.5e} max {torch.max(model.blocks[i].LT.sigma).item():.5e}") #
            # print(i,torch.mean(model.blocks[i].LT.gate).item(),torch.min(model.blocks[i].LT.gate).item(),torch.max(model.blocks[i].LT.gate).item())

    if scheduler is not None and args['training']['scheduler']['name'] != 'ReduceLROnPlateau':
        scheduler.step()

    # collect all process losses
    if args["distributed"]:
        loss_tensors = []
        for loss_component in loss_history:
            avg_loss = np.mean(loss_component)
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensors.append(loss_tensor / args.world_size)
        
        train_losses = [tensor.item() for tensor in loss_tensors]
        #Wasted
    else:
        train_losses = [np.sum(loss_component)/len(train_dataset) for loss_component in loss_history]

    return train_losses

# Do an evaluation on the test/validate dataset
def evaluate_process(dataloader,save_path=None,detail=False):
    model.eval()

    if args["distributed"]:
        dataloader.sampler.set_epoch(epoch)

    with torch.no_grad():

        for data in dataloader:
            # Apply the model
            preds = model(data)
            
            labels=data['labels'].to(device)
            loss_result =myloss(preds,labels,mask=data.get('mask', None))

            if isinstance(loss_result, (list, tuple)): #multiple losses
                loss_values = [loss_item.item() if hasattr(loss_item, 'item') else loss_item for loss_item in loss_result]
            else: #simgle loss                                                           
                loss_values = [loss_result.item()]
            
            if 'loss_history' not in locals():
                loss_history = [[] for _ in range(len(loss_values))]
            
            for i, loss_val in enumerate(loss_values):
                loss_history[i].append(loss_val) #*data['labels'].shape[0]

    if save_path:
        plot_loss_distribution(loss_history[args["training"]['info_index']],save_path)

    # collect all process losses
    if args["distributed"]:
        loss_tensors = []
        for loss_component in loss_history:
            avg_loss = np.mean(loss_component)
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensors.append(loss_tensor / args.world_size)
        
        train_losses = [tensor.item() for tensor in loss_tensors]
    else:
        train_losses = [np.sum(loss_component)/len(dataloader.dataset) for loss_component in loss_history]

    if detail:
        return train_losses,loss_history
    return train_losses

# Training loop
if not args["evaluate"]:
    if args['system']['verbose']:
        logger("Start training...")

    best_epoch = 0
    best_loss = None
    counter = 0

    start_time = time.time()
    for epoch in range(n_epoch):
        epoch_start_time = time.time()
        
        lr=optimizer.param_groups[0]["lr"]
        if lr<1e-6:
            break
        train_loss = train_process(epoch)

        if args['training']['scheduler']['name'] == 'ReduceLROnPlateau' :
            val_loss=evaluate_process(valid_loader)
            # print(val_loss)
            scheduler.step(val_loss[0])

        # time cost (total time per epoch and estimated remaining time)
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = n_epoch - epoch - 1
        estimated_remaining_time = avg_time_per_epoch * remaining_epochs

        if args['system']['verbose']:
            output = "Epoch {} | lr: {:.6f} | training loss: ".format(epoch, lr)
            output+=utils.output_loss_(train_loss, args["training"]["loss_meaning"])
            output += " | epoch time: {:.2f}s".format(epoch_time)
            output += " | remaining: {}".format(utils.format_time(estimated_remaining_time))
            logger(output)

        # save checkpoint if needed (ATTENTION: including model, optimizer, scheduler states)
        if args['training']['saving_checkpoint'] and (epoch + 1) % args['training']['checkpoint_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_record': train_loss,
                }, os.path.join(args['saving_path'], "checkpoint_{}_{:.4f}.pth".format(epoch,train_loss[0])))
            model=model.to(device)
            if args['system']['verbose']:
                logger("Epoch {} | save checkpoint in {}".format(epoch, args['saving_path']))
        
        if args['training']['train_info'] and (epoch + 1) % args['training']['info_freq'] == 0:
            # Compute std for train, val and test
            # this is very slow
            train_loss,train_loss_list=evaluate_process(train_loader_info,detail=True)
            val_loss,val_loss_list=evaluate_process(valid_loader_info,detail=True)
            test_loss,test_loss_list=evaluate_process(test_loader_info,detail=True)
            index=args["training"]['info_index']
            train_loss_std=np.std(train_loss_list[index])
            val_loss_std=np.std(val_loss_list[index])
            test_loss_std=np.std(test_loss_list[index])
            logger("Epoch {} | train loss: {:.4f} | val loss: {:.4f} | test loss: {:.4f} | train loss std: {:.4f} | val loss std: {:.4f} | test loss std: {:.4f}"
                   .format(epoch,train_loss[index],val_loss[index],test_loss[index],train_loss_std,val_loss_std,test_loss_std))

        # evaluate on validation set 
        if (epoch + 1) % args['training']['eval_freq'] == 0:
            val_loss = evaluate_process(valid_loader)
            test_loss= evaluate_process(test_loader)
            if args['system']['verbose']:
                output="Epoch {} | validation loss: ".format(epoch)
                output+=utils.output_loss_(val_loss, args["training"]["loss_meaning"])
                logger(output)
                output="Epoch {} | test loss: ".format(epoch)
                output+=utils.output_loss_(test_loss, args["training"]["loss_meaning"])
                logger(output)
            
            # save the best model on validation set 
            index=args["training"]['best_index']
            if index==-1:
                continue
            if not best_loss or val_loss[index] < best_loss[index]:
                counter = 0
                best_epoch = epoch
                best_loss = val_loss 
                torch.save(model.cpu().state_dict(), os.path.join(args['saving_path'], "best_model.pth")) 
                model=model.to(device)
                if args['system']['verbose']:
                    logger("Epoch {} | save best models in {}".format(epoch, args['saving_path']))
            # early stopping
            elif args['training']['patience'] != -1:
                counter += 1
                if counter >= args['training']['patience']:
                    if args['system']['verbose']:
                        logger("Early stop at epoch {}".format(epoch))
                    break

    logger("Training Finished!")
    
    # load best model on validation set or last model
    if not best_loss:
        torch.save(model.cpu().state_dict(), os.path.join(args['saving_path'], "best_model.pth"))
    else:
        model.load_state_dict(torch.load(os.path.join(args['saving_path'], "best_model.pth")))
        logger("Load best models at epoch {} from {}".format(best_epoch, args['saving_path']))        
    model=model.to(device)
    
    valid_loss_record = evaluate_process(valid_loader)
    if args['system']['verbose']:
        output="Validation loss: "
        output += utils.output_loss_(valid_loss_record, args["training"]["loss_meaning"])
        logger(output)

# valuate on test set
test_loss_record = evaluate_process(test_loader_info,save_path=os.path.join(args['saving_path'], "test_loss_distribution.png"))

output="Test loss: "
output += utils.output_loss_(test_loss_record, args["training"]["loss_meaning"])

if args['system']['verbose']:
    logger(output)
else :
    print(output)

if args["distributed"]:
    dist.destroy_process_group()
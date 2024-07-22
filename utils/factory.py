

import numpy as np
import torch
import random
from tqdm import tqdm


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id, rank=0, seed=11):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        if name not in list(own_state.keys()) or 'output_conv' in name:
            ckpt_name.append(name)
            continue
        own_state[name].copy_(param)
        cnt += 1
    tqdm.write('reused param: {}'.format(cnt))
    return model


def get_optimizer(net, args):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def get_scheduler(optimizer, args, iters_per_epoch):
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max * iters_per_epoch,
            eta_min=args.eta_min,
        )
    elif args.scheduler == 'multi':
        milestones = [epoch * iters_per_epoch for epoch in args.steps]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma,
        )
    else:
        raise NotImplementedError
    return scheduler
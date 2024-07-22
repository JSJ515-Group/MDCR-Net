import sys
from functools import partial
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
import os
import torch
from torch.utils.data import DataLoader
import configs.culane
import train
from dataset.CULane import MyLaneAugDataSet
from model.net import Net
from utils.factory import get_optimizer, seed_everything, worker_init_fn, load_my_state_dict
from utils.my_log import get_logger, Logger
from utils.save_model import save_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    # Add arguments to the parser
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--dataset', type=str, default=configs.culane.dataset)
    parser.add_argument('--data_root', type=str, default=configs.culane.data_root)
    parser.add_argument('--num_class', type=int, default=configs.culane.num_class)
    parser.add_argument('--optimizer', type=str, default=configs.culane.optimizer)
    parser.add_argument('--scheduler', type=str, default=configs.culane.scheduler)
    parser.add_argument('--log_dir', type=str, default=configs.culane.log_dir)
    parser.add_argument('--epoch', type=int, default=configs.culane.epoch)
    parser.add_argument('--device', type=str, default=configs.culane.device)
    parser.add_argument('--num_workers', type=int, default=configs.culane.num_workers)
    parser.add_argument('--T_max', type=int, default=configs.culane.T_max)
    parser.add_argument('--train_batch_size', type=int, default=configs.culane.train_batch_size)
    parser.add_argument('--valid_batch_size', type=int, default=configs.culane.valid_batch_size)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument('--lr', type=float, default=configs.culane.lr)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--eta_min', type=float, default=0)
    parser.add_argument('--steps', type=list, default=configs.culane.steps)
    parser.add_argument('--gamma', type=float, default=0.1)
    return parser


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    sys.stdout = Logger(folder_path=args.log_dir, file_name='log_file.log')
    print(args)
    seed_everything(seed=11)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Current device: {torch.cuda.get_device_name(device)}, device index: {args.device}")
    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU.")
    # pre_train_culane
    train_dataset = MyLaneAugDataSet(dataset_path=args.data_root, data_list='train_gt',
                                     mode='train')
    # val_gt_121
    val_dataset = MyLaneAugDataSet(dataset_path=args.data_root, data_list='val_gt_121',
                                   mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True,
                              worker_init_fn=partial(worker_init_fn, rank=0, seed=11), prefetch_factor=2,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.valid_batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True, pin_memory=True,
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=11), prefetch_factor=2,
                            persistent_workers=True)

    net = Net(num_classes=5).to(device, non_blocking=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    # print(net)

    checkpoint = torch.load('')
    net = load_my_state_dict(net, checkpoint['model'])

    optimizer = get_optimizer(net, args)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_miou = 0
    logger = get_logger(args.log_dir, args)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        path_checkpoint = os.path.join(args.log_dir, 'current.pth')
        checkpoint = torch.load(path_checkpoint)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        best_miou = torch.load(os.path.join(args.log_dir, 'best_miou.pth'))['miou']
        best_precision = torch.load(os.path.join(args.log_dir, 'best_precision.pth'))['precision']
        best_recall = torch.load(os.path.join(args.log_dir, 'best_recall.pth'))['recall']
        best_f1 = torch.load(os.path.join(args.log_dir, 'best_f1_score.pth'))['f1_score']
        
        tqdm.write(f"==> Resuming training from epoch {resume_epoch}")
    else:
        resume_epoch = 0

    current_model_path = os.path.join(args.log_dir, 'current.pth')
    best_model_miou_path = os.path.join(args.log_dir, 'best_miou.pth')
    best_model_precision_path = os.path.join(args.log_dir, 'best_precision.pth')
    best_model_recall_path = os.path.join(args.log_dir, 'best_recall.pth')
    best_model_f1_score_path = os.path.join(args.log_dir, 'best_f1_score.pth')
    for epoch in range(resume_epoch, args.epoch):
        train_dataset.epoch = epoch
        print(f"Best F1: {best_f1*100:.2f}%, Best miou: {best_miou*100:.2f}%, Best recall: {best_recall*100:.2f}%, Best precision: {best_precision*100:.2f}%")

        train.train(net, train_loader, optimizer, scheduler, epoch, logger, scaler=scaler)
        miou, precision, recall, f1_score, val_loss = train.valid(net, val_loader, epoch, optimizer, logger)

        scheduler.step(val_loss)

        # 保存最佳模型权重
        if miou > best_miou:
            best_miou = miou
            save_model(net, optimizer, scheduler, epoch, miou, precision, recall, f1_score, args,
                       best_model_miou_path, scaler)

        if precision > best_precision:
            best_precision = precision
            save_model(net, optimizer, scheduler, epoch, miou, precision, recall, f1_score, args,
                       best_model_precision_path, scaler)

        if recall > best_recall:
            best_recall = recall
            save_model(net, optimizer, scheduler, epoch, miou, precision, recall, f1_score, args,
                       best_model_recall_path, scaler)

        if f1_score > best_f1:
            best_f1 = f1_score
            save_model(net, optimizer, scheduler, epoch, miou, precision, recall, f1_score, args,
                       best_model_f1_score_path, scaler)

        # 保存当前模型权重
        save_model(net, optimizer, scheduler, epoch, miou, precision, recall, f1_score, args,
                   current_model_path, scaler)

    logger.close()
    sys.exit()

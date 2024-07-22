
import torch


def save_model(net, optimizer, scheduler, epoch, miou, precision, recall, f1_score, args, current_model_path,
               scaler=None):
    state = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'miou': miou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }
    if args.amp:
        state["scaler"] = scaler.state_dict() if scaler is not None else None

    torch.save(state, current_model_path)



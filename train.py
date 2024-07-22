import time
import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
from utils.metrics import get_metrics
import configs.culane


def valid(net, data_loader, epoch, optimizer, logger=None):
    net.eval()
    start_time = time.time()
    avg_loss_total = 0
    avg_loss_att = 0
    avg_loss_seg = 0
    avg_loss_aux_2 = 0
    avg_loss_aux_3 = 0
    avg_iou = 0
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    precision = 0
    recall = 0
    f1_score = 0
    CEL = torch.nn.CrossEntropyLoss(weight=class_weights)
    MSEL = torch.nn.MSELoss()

    with torch.no_grad():
        bar = tqdm(data_loader, leave=False)
        for b_idx, (input, seg_target_2D, seg_target_3D, exist, names) in enumerate(bar):
            img, seg_label_2D, seg_target_3D, lane_label = input.cuda(non_blocking=True), seg_target_2D.cuda(
                non_blocking=True), seg_target_3D.cuda(non_blocking=True), exist.float().cuda(non_blocking=True)
            seg_out, att_map_1, att_map_2, aux_2, aux_3 = net(img)

            seg_out_3D = (torch.softmax(seg_out, dim=1))[:, 1:, :, :]
            att_out_1 = seg_out_3D * att_map_1
            att_target_1 = seg_target_3D * att_map_1

            att_out_2 = seg_out_3D * att_map_2
            att_target_2 = seg_target_3D * att_map_2

            loss_att_1 = MSEL(att_out_1, att_target_1) + torch.abs(
                torch.mean(att_target_1) - 0.6 * torch.mean(seg_target_3D))
            loss_att_2 = MSEL(att_out_2, att_target_2) + torch.abs(
                torch.mean(att_target_2) - 0.6 * torch.mean(seg_target_3D))
            loss_att = loss_att_1 + loss_att_2
            loss_seg = CEL(seg_out, seg_label_2D)
            loss_aux_2 = CEL(aux_2, seg_label_2D)
            loss_aux_3 = CEL(aux_3, seg_label_2D)
            # loss = loss_att * 50 + loss_seg + loss_aux_1 * 0.75 + loss_aux_2 * 0.75 + loss_aux_3 * 0.75
            loss = loss_att * 50 + loss_seg + loss_aux_2 * 0.7 + loss_aux_3 * 0.3

            seg_out_2D = torch.argmax(seg_out, dim=1)
            res = get_metrics(seg_out_2D, seg_label_2D, num_classes=5)
            iou = res[0]
            recall += res[1]
            precision += res[2]
            f1_score += res[3]

            avg_loss_total += loss.item()
            avg_loss_att += loss_att.item()
            avg_loss_seg += loss_seg.item()
            avg_loss_aux_2 += loss_aux_2.item()
            avg_loss_aux_3 += loss_aux_3.item()
            avg_iou += iou

        avg_loss_total = avg_loss_total / len(data_loader)
        avg_loss_seg = avg_loss_seg / len(data_loader)
        avg_loss_aux_2 = avg_loss_aux_2 / len(data_loader)
        avg_loss_aux_3 = avg_loss_aux_3 / len(data_loader)
        avg_loss_att = avg_loss_att / len(data_loader)
        avg_iou = avg_iou / len(data_loader)
        precision /= len(data_loader)
        recall /= len(data_loader)
        f1_score /= len(data_loader)

        if logger is not None:
            # Write the validation loss, mIOU, precision, recall, accuracy, and F1-score to TensorBoard
            logger.add_scalar('Validation/Loss Total', avg_loss_total, epoch)
            logger.add_scalar('Validation/Loss Seg', avg_loss_seg, epoch)
            logger.add_scalar('Validation/Loss Aux_2', avg_loss_aux_2, epoch)
            logger.add_scalar('Validation/Loss Aux_3', avg_loss_aux_3, epoch)
            logger.add_scalar('Validation/Loss Att', avg_loss_att, epoch)
            logger.add_scalar('Validation/mIOU', avg_iou, epoch)
            logger.add_scalar('Validation/Precision', precision, epoch)
            logger.add_scalar('Validation/Recall', recall, epoch)
            logger.add_scalar('Validation/F1-Score', f1_score, epoch)
            logger.add_scalar('Validation/Lr', optimizer.param_groups[-1]['lr'], epoch)

        end_time = time.time()
        execution_time = end_time - start_time

        print((
            'Validation: lr: {lr:.5f}\t' 'Loss total: {loss_t:.4f}\t'
            'mIOU: {avg_iou:.4f}\t' 'Precision: {precision:.4f}\t' 'Recall: {recall:.4f}\t' 'F1_score: {f1_score:.4f}\t'
            'Execution Time: {execution_time:.2f} seconds'
            .format(loss_t=avg_loss_total,
                    avg_iou=avg_iou,
                    lr=optimizer.param_groups[-1]['lr'],
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    execution_time=execution_time)))
    return avg_iou, precision, recall, f1_score, avg_loss_total


def train(net, data_loader, optimizer, scheduler, epoch, logger=None, scaler=None):
    net.train()
    avg_loss_total = 0
    avg_loss_att = 0
    avg_loss_seg = 0
    avg_iou = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1_score = 0
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    CEL = torch.nn.CrossEntropyLoss(weight=class_weights)
    MSEL = torch.nn.MSELoss()

    print_freq = configs.culane.print_freq
    total_steps = len(data_loader)
    train_progress_bar = tqdm(total=total_steps, desc=f'Train Epoch [{epoch}/{configs.culane.epoch}]', unit='it')
    for b_idx, (input, seg_target_2D, seg_target_3D, exist, names) in enumerate(data_loader):
        global_step = epoch * len(data_loader) + b_idx
        img, seg_label_2D, seg_target_3D, lane_label = input.cuda(non_blocking=True), seg_target_2D.cuda(
            non_blocking=True), seg_target_3D.cuda(non_blocking=True), exist.float().cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            seg_out, att_map_1, att_map_2, aux_2, aux_3 = net(img)

            seg_out_3D = (torch.softmax(seg_out, dim=1))[:, 1:, :, :]
            att_out_1 = seg_out_3D * att_map_1
            att_target_1 = seg_target_3D * att_map_1

            att_out_2 = seg_out_3D * att_map_2
            att_target_2 = seg_target_3D * att_map_2

            loss_att_1 = MSEL(att_out_1, att_target_1) + torch.abs(
                torch.mean(att_target_1) - 0.6 * torch.mean(seg_target_3D))
            loss_att_2 = MSEL(att_out_2, att_target_2) + torch.abs(
                torch.mean(att_target_2) - 0.6 * torch.mean(seg_target_3D))
            loss_att = loss_att_1 + loss_att_2
            loss_seg = CEL(seg_out, seg_label_2D)
            loss_aux_2 = CEL(aux_2, seg_label_2D)
            loss_aux_3 = CEL(aux_3, seg_label_2D)
            # loss = loss_att * 50 + loss_seg + loss_aux_1 * 0.75 + loss_aux_2 * 0.75 + loss_aux_3 * 0.75
            loss = loss_att * 50 + loss_seg + loss_aux_2 * 0.7 + loss_aux_3 * 0.3

            with torch.no_grad():
                seg_out_2D = torch.argmax(seg_out, dim=1)
                res = get_metrics(seg_out_2D, seg_label_2D, num_classes=5)
                iou = res[0]
                recall = res[1]
                precision = res[2]
                f1_score = res[3]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # scheduler.step()

        avg_loss_total += loss.item()
        avg_loss_att += loss_att.item()
        avg_loss_seg += loss_seg.item()
        avg_iou += iou
        avg_recall += recall
        avg_precision += precision
        avg_f1_score += f1_score

        if logger is not None:
            logger.add_scalar('Train Loss/Total', loss.item(), global_step)
            logger.add_scalar('Train Loss/Seg', loss_seg.item(), global_step)
            logger.add_scalar('Train Loss/Aux_2', loss_aux_2.item(), global_step)
            logger.add_scalar('Train Loss/Aux_3', loss_aux_3.item(), global_step)
            logger.add_scalar('Train Loss/Att', loss_att.item(), global_step)
            logger.add_scalar('Train/mIOU', iou, global_step)
            logger.add_scalar('Train/Precision', precision, global_step)
            logger.add_scalar('Train/Recall', recall, global_step)
            logger.add_scalar('Train/F1-Score', f1_score, global_step)
            logger.add_scalar('Train/Lr', optimizer.param_groups[-1]['lr'], global_step)

        if (b_idx + 1) % print_freq == 0:
            avg_loss_total = avg_loss_total / print_freq
            avg_loss_att = avg_loss_att / print_freq
            avg_loss_seg = avg_loss_seg / print_freq
            avg_iou = avg_iou / print_freq
            avg_precision = avg_precision / print_freq
            avg_recall = avg_recall / print_freq
            avg_f1_score = avg_f1_score / print_freq
            train_progress_bar.set_postfix(
                {'Lr': optimizer.param_groups[-1]['lr'],
                 'Loss_total': avg_loss_total,
                 'Loss_seg': avg_loss_seg,
                 'Loss_att': avg_loss_att,
                 'mIOU': avg_iou,
                 'Precision': avg_precision,
                 'Recall': avg_recall,
                 'F1-Score': avg_f1_score})
            train_progress_bar.update(print_freq)
            avg_loss_total = 0
            avg_loss_att = 0
            avg_loss_seg = 0
            avg_iou = 0
            avg_precision = 0
            avg_recall = 0
            avg_f1_score = 0
    train_progress_bar.close()

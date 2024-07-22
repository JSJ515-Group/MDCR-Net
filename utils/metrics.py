
import numpy as np
import configs.culane


def get_metrics(pred, label, num_classes=configs.culane.num_class + 1):
    pred = pred.cpu().data.numpy().flatten()
    label = label.cpu().data.numpy().flatten()

    hist = np.bincount(num_classes * label.astype(int) + pred, minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)

    iou = np.diag(hist) / (np.sum(hist, axis=1) + np.sum(hist, axis=0) - np.diag(hist) + 1e-8)
    miou = np.nanmean(iou)
    recall = np.diag(hist) / (np.sum(hist, axis=1) + 1e-8)
    recall = np.nanmean(recall)
    precision = np.diag(hist) / (np.sum(hist, axis=0) + 1e-8)
    precision = np.nanmean(precision)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_score = np.nanmean(f1_score)
    res = [miou, recall, precision, f1_score]
    return res



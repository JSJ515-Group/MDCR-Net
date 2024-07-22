import os
from torch.utils.data import Dataset
from dataset.mytransforms import *
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TusimpleDataSet(Dataset):
    def __init__(self, dataset_path='/home/zx/DataSet/Tusimple/seg_label/list', data_list='train_gt',
                 mode='train', epoch=0):

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.epoch = epoch
            self.mode = mode
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('seg_label/list', '') + line.strip().split(" ")[0])
                self.label_list.append(dataset_path.replace('seg_label/list', '') + line.strip().split(" ")[1])
                self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]),
                                                 int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5]),
                                                 int(line.strip().split(" ")[6]), int(line.strip().split(" ")[7])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path

        self.segment_transform = transforms.Compose([
            FreeScaleMask((288, 800)),
            MaskToTensor(),
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 对图像和标签随机进行旋转和左右平移
        self.simu_transform = Compose2([
            RandomRotate(6),
            RandomUDoffsetLABEL(100),
            RandomLROffsetLABEL(200)
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])
        label = Image.open(self.label_list[idx])

        if self.mode == 'train':
            image, label = self.simu_transform(image, label)

        elif self.mode == 'strong_train':
            image, label = self.simu_transform(image, label)
            # 增加数据增强，提升模型的泛化能力
            image, label = hflip(image, label, p=0.5)  # 进行随机翻转
            image = color_jitter(image, p=0.5)  # 进行颜色抖动处理
            image = random_grayscale(image, p=0.5)  # 进行灰度化处理，但是这个不会改变通道数
            image = blur(image, p=0.5)  # 进行高斯模糊

        exist = self.exist_list[idx]
        image = self.img_transform(image)
        # 此时label_2D里全是0，1，2，3，4，其中0表示背景，1，2，3，4表示第几条车道线
        label_2D = self.segment_transform(label)
        label_2D = label_2D[:, :, 0]
        np_label = np.array(label_2D)

        lanes = []
        for i in range(1, 7):
            lane = np.where(np_label == i, 1, 0)
            lane = np.expand_dims(lane, axis=2)
            lanes.append(lane)
        out = np.concatenate(lanes, axis=2)
        label_3D = self.test_transform(out).float()

        return image, label_2D, label_3D, exist, self.img[idx]


# --------------------------------------------------------------------------------------------------------------------#

#                                 对读取的数据集进行可视化，判断是否正确读取

# --------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    train_dataset = TusimpleDataSet(dataset_path='/home/zx/DataSet/Tusimple/seg_label/list', data_list='train_val_gt',
                                    mode='train')
    test_dataset = TusimpleDataSet(dataset_path='/home/zx/DataSet/Tusimple/seg_label/list', data_list='test_gt',
                                   mode='test')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    num_images = 4  # 要显示的图像数量
    for i, (input, target, cls_label, exist, names) in enumerate(test_loader):
        print(names[i])  # 输出图像路径
        print(cls_label.size())  # 输出图像尺寸
        cls_label = (cls_label.numpy())[0]

        input = (input.numpy())[0]
        # 将input重新映射到[0, 255]范围
        input_min = np.min(input)
        input_max = np.max(input)
        input = (input - input_min) / (input_max - input_min) * 255
        # 将像素值限制在[0, 255]范围内
        input = np.clip(input, 0, 255)
        # 转换为正确的图像格式
        input = np.transpose(input, (1, 2, 0)).astype('uint8')

        target = (target.numpy())[0]

        for row in range(cls_label.shape[0]):
            line = cls_label[row]
            buf = 0
            for x in range(len(line) - 2):
                if line[x].all() != 0:
                    loc = line[x]
                    buf = loc + buf
                    input = cv2.circle(input, (int(buf * 800), row * 8), 3, (255, 255, 0), -1)

        plt.figure(1)
        plt.subplot(num_images, 2, i * 2 + 1)
        plt.imshow(input)
        plt.title(names[i])

        plt.subplot(num_images, 2, i * 2 + 2)
        plt.imshow(target)
        plt.title(names[i])

        if i == num_images - 1:
            break
    plt.show()

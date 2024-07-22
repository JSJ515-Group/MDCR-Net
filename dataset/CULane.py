import warnings
import configs.culane
import multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataset.mytransforms import *
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class PreLaneDataSet(Dataset):
    def __init__(self, dataset_path='/home/zx/DataSet/CULane/list/test_split', data_list='test0_normal',
                 transform=None):

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('list/test_split', '') + line.strip().split(" ")[0])
                self.label_list = ['E:/DataSet/CULane' + filename.replace('E:/DataSet/CULane', '')
                                   for filename in self.img_list]

        self.img_path = dataset_path
        self.transform = transform

        self.segment_transform = transforms.Compose([
            FreeScaleMask((288, 800)),
            MaskToTensor(),
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.origi_img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
        ])

        self.simu_transform = Compose2([
            RandomRotate(6),
            RandomUDoffsetLABEL(100),
            RandomLROffsetLABEL(200)
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])

        if self.transform:
            original_img = image.copy()
            original_img = self.origi_img_transform(original_img)

            image = self.img_transform(image)

        return original_img, image, self.img[idx]


class MyLaneAugDataSet(Dataset):
    def __init__(self, dataset_path='E:/DataSet/CULane/list', data_list='train_gt', mode='train', epoch=100):
        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.epoch = epoch
            self.mode = mode
            self.img = []
            self.img_list = []
            self.label_list = []
            self.exist_list = []

            for line in f:
                line_data = line.strip().split(" ")
                self.img.append(line_data[0])
                self.img_list.append(dataset_path.replace('/list', '') + line_data[0])
                self.label_list.append(dataset_path.replace('/list', '') + line_data[1])
                self.exist_list.append(
                    np.array([int(line_data[2]), int(line_data[3]), int(line_data[4]), int(line_data[5])]))

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

        self.simu_transform = Compose2([
            RandomRotate(6),
            RandomUDoffsetLABEL(100),
            RandomLROffsetLABEL(200),
            # RandomScaleAndPad()
        ])

    def __len__(self):
        return len(self.img_list)

    def preprocess(self, img, label, exist=None):

        if self.mode == 'train':
            img, label = self.simu_transform(img, label)

        if self.mode == 'train_strong':
            img, label = self.simu_transform(img, label)
            img, label = hflip(img, label, p=0.5)
            img = color_jitter(img, p=0.5)
            img = random_grayscale(img, p=0.5)
            img = blur(img, p=0.5)
            img = lanes_add(img, label)

        img = self.img_transform(img)
        label_2D = self.segment_transform(label)
        return img, label_2D, exist

    def __getitem__(self, idx):

        img_path = self.img_list[idx]
        label_path = self.label_list[idx]

        img = Image.open(img_path)
        label = Image.open(label_path)

        exist = self.exist_list[idx]

        img, label_2D, exist = self.preprocess(img, label, exist)

        np_label = np.array(label_2D)

        lanes = []
        for i in range(1, 5):
            lane = np.where(np_label == i, 1, 0)
            lane = np.expand_dims(lane, axis=2)
            lanes.append(lane)

        out = np.concatenate(lanes, axis=2)
        label_3D = self.test_transform(out).float()

        return img, label_2D, label_3D, exist, img_path


def lanes_add(image, label_image):
    image_np = np.array(image)
    label_image_np = np.array(label_image)

    white_mask = cv2.cvtColor(label_image_np, cv2.COLOR_GRAY2BGR)
    white_mask[label_image_np > 0] = (255, 255, 255)

    alpha = np.random.uniform(0.5, 0.9)
    beta = np.random.uniform(0, 0.1)
    gamma = np.random.uniform(-50, 150)

    result_frame = cv2.addWeighted(image_np, alpha, white_mask, beta, gamma)

    result_frame = Image.fromarray(result_frame)

    return result_frame


if __name__ == "__main__":
    train_dataset = MyLaneAugDataSet(dataset_path='/home/zx/night/list', data_list='train_gt', mode='train')
    val_dataset = MyLaneAugDataSet(dataset_path='/home/zx/DataSet/data/list', data_list='val_gt', mode='val')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)

    num_images = 4
    for i, (input, target, cls_label, exist, names) in enumerate(train_loader):
        print(names[i])
        print(cls_label.size())
        cls_label = (cls_label.numpy())[0]

        input = (input.numpy())[0]

        input_min = np.min(input)
        input_max = np.max(input)
        input = (input - input_min) / (input_max - input_min) * 255

        input = np.clip(input, 0, 255)
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

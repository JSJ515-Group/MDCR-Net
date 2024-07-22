import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from net_1.conv_mix_att_11_spa_flip_aux import Vis_Net_att_spa_flip_aux
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


# --------------------------------------------------------------------------------------------------------------------#

#                                  该方法用于关闭窗口，空格键关闭

# --------------------------------------------------------------------------------------------------------------------#
def on_key_press(event):
    if event.key == ' ':
        plt.close()


# --------------------------------------------------------------------------------------------------------------------#

#                                    数据处理，应保持与模型处理方法一致

# --------------------------------------------------------------------------------------------------------------------#
data_transform = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# --------------------------------------------------------------------------------------------------------------------#

#                                     加载训练完成的模型，并载入权重

# --------------------------------------------------------------------------------------------------------------------#
# create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Vis_Net_att_spa_flip_aux(num_classes=5)
# model = torch.nn.DataParallel(model)
model.to(device)
# load model weights
checkpoint = torch.load('/home/zx/LandeCode/EndNet/mylogs/Net_att_spa_flip_aux_train_v2/best_f1_score.pth')
# model = load_my_state_dict(model, checkpoint['model'])
model.load_state_dict(checkpoint['model'], strict=False)
# print(model)


# --------------------------------------------------------------------------------------------------------------------#

#                                       可视化模型权重的直方图

# --------------------------------------------------------------------------------------------------------------------#
# weights_keys = model.state_dict().keys()
# # print(weights_keys)
# for key in weights_keys:
#     # remove num_batches_tracked para(in bn)
#     if "num_batches_tracked" in key:
#         continue
#     # [kernel_number, kernel_channel, kernel_height, kernel_width]
#     weight_t = model.state_dict()[key].cpu().numpy()
#
#     # read a kernel information
#     # k = weight_t[0, :, :, :]
#
#     # calculate mean, std, min, max
#     weight_mean = weight_t.mean()
#     weight_std = weight_t.std(ddof=1)
#     weight_min = weight_t.min()
#     weight_max = weight_t.max()
#     print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
#                                                                weight_std,
#                                                                weight_max,
#                                                                weight_min))
#     # plot hist image
#     # plt.close()
#     fig = plt.figure()
#     weight_vec = np.reshape(weight_t, [-1])
#     plt.hist(weight_vec, bins=50)
#     plt.title(key)
#     plt.connect('key_press_event', on_key_press)  # 空格键关闭窗口
#     fig.canvas.manager.full_screen_toggle() # 设置窗口为全屏
#     plt.show()


# --------------------------------------------------------------------------------------------------------------------#

#                                       可视化模型中间层输出

# --------------------------------------------------------------------------------------------------------------------#
# load image
img = Image.open("00000.jpg")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
img = img.cuda()

# forward
_, _, _, _, _, _, out_put = model(img)
count = 0
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().cpu().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
    count = count + 1
    # show top 12 feature maps
    fig = plt.figure()
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # [H, W, C]
        plt.imshow(im[:, :, i])
        plt.title("conv {} vision".format(count))
    plt.connect('key_press_event', on_key_press)  # 空格键关闭窗口
    fig.canvas.manager.full_screen_toggle()  # 设置窗口为全屏
    plt.savefig('conv{}.png'.format(count))
    plt.show()

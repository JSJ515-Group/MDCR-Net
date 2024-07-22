dataset = 'Tusimple'
# data_root = '/home/zx/DataSet/Tusimple/seg_label/list'
data_root = '/mnt/sdb/A515/zhangxu/Tusimple/seg_label/list'
num_class = 6
optimizer = 'SGD'  # Adam、SGD
scheduler = 'multi'  # cos、multi
log_dir = 'logs/Net_full_exist'
epoch = 120
print_freq = 1
device = '0'
train_batch_size = 8
valid_batch_size = 8
num_workers = 8

lr = 0.01
T_max = 100

warmup = 'linear'
steps = [2, 4]

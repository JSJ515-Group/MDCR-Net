dataset = 'CULane'
data_root = '/home/zx/night/list'
num_class = 4
optimizer = 'SGD'  # Adam、SGD
scheduler = 'cos'  # cos、multi

# model_3
log_dir = 'mylogs/SPA-6'

epoch = 120
print_freq = 1
device = '1'
train_batch_size = 8
valid_batch_size = 8
num_workers = 2

lr = 0.01
T_max = 120

warmup = 'linear'
steps = [2, 4]

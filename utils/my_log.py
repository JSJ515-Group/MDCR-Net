
import os
import sys
from torch.utils.tensorboard import SummaryWriter


class DistSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super(DistSummaryWriter, self).__init__(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        super(DistSummaryWriter, self).add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        super(DistSummaryWriter, self).add_figure(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        super(DistSummaryWriter, self).add_graph(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        super(DistSummaryWriter, self).add_histogram(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        super(DistSummaryWriter, self).add_image(*args, **kwargs)

    def close(self):
        super(DistSummaryWriter, self).close()



def get_logger(work_dir, cfg):
    logger = DistSummaryWriter(work_dir)
    config_txt = os.path.join(work_dir, 'cfg.log')
    with open(config_txt, 'w') as fp:
        for arg in vars(cfg):
            fp.write(str(f"    {arg}: {getattr(cfg, arg)} \n"))
    return logger


class Logger(object):
    def __init__(self, folder_path="logs", file_name="Default.log"):
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)

        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

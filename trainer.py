import os
import os.path as osp
import time
import torch
import datetime
import logging
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from verifier import Verifier
from networks import get_model
from utils import *
from criterion import *
from lr_scheduler import WarmupPolyLR

from torch.utils.tensorboard import SummaryWriter


__BA__ = ["CE2P", "FaceParseNet18", "FaceParseNet34", "FaceParseNet50", "FaceParseNet"]


class Trainer(object):
    """Training pipline"""

    def __init__(self, data_loader, config, val_loader):

        # Data loader
        self.data_loader = data_loader
        self.verifier = Verifier(val_loader, config)
        self.writer = SummaryWriter('runs/training/{}'.format(config.arch))

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel
        self.arch = config.arch

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.total_iters = self.epochs * len(self.data_loader)

        self.classes = config.classes
        self.g_lr = config.g_lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.pretrained_model = config.pretrained_model  # int type
        self.indicator = False if self.pretrained_model > 0 else True  # 是否是断点状态？

        self.img_path = config.img_path
        self.label_path = config.label_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.sample_step = config.sample_step
        self.tb_step = config.tb_step

        # Path
        self.sample_path = osp.join(config.sample_path, self.arch)
        self.model_save_path = osp.join(
            config.model_save_path, self.arch)

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        self.lr_scheduler = WarmupPolyLR(
            self.g_optimizer, max_iters=self.total_iters, power=0.9, warmup_factor=1.0 / 3, warmup_iters=500,
            warmup_method='linear')

    def train(self):
        # Get current date and time
        now = datetime.datetime.now()
        # Format filename with current date and time
        filename = 'train{}_{}.log'.format(self.arch, now.strftime('%Y%m%d_%H-%M-%S'))
        # Configure logging
        logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # 创建一个StreamHandler，用于将日志输出到终端
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        # 获取根logger，并添加StreamHandler
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
        root_logger.info("Start training...")
        # 记录开始训练时间
        start_time1 = time.time()
        start_time2 = time.time()
        # 如果是断点状态，则从断点处开始训练
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        criterion = CriterionAll()
        criterion.cuda()
        best_miou = 0

        # Data iterator
        progress_bar = tqdm(range(start, self.epochs), desc='Epoch-Check', mininterval=10, maxinterval=60, ncols=100)
        for epoch in progress_bar:
            self.G.train()
            for i_iter, batch in enumerate(self.data_loader):
                # 记录网络结构到tensorboard
                if epoch == 0:
                    self.writer.add_graph(self.G, batch[0].cuda())
                i_iter += len(self.data_loader) * epoch
                # lr = adjust_learning_rate(self.g_lr,
                #                           self.g_optimizer, i_iter, self.total_iters)

                imgs, labels, edges = batch
                size = labels.size()
                imgs = imgs.cuda()
                labels = labels.cuda()

                if self.arch in __BA__:
                    edges = edges.cuda()
                    preds = self.G(imgs)
                    c_loss = criterion(preds, [labels, edges])

                    # labels_predict is a tensor
                    labels_predict = preds[0][-1]

                else:
                    labels = labels.cuda()
                    # oneHot_size = (size[0], self.classes, size[1], size[2])
                    # labels_real = torch.cuda.FloatTensor(
                    #     torch.Size(oneHot_size)).zero_()
                    # labels_real = labels_real.scatter_(
                    #     1, labels.data.long().cuda(), 1.0)

                    labels_predict = self.G(imgs)
                    # 打印labels_predict的类型
                    # print(type(labels_predict))
                    c_loss = cross_entropy2d(
                        labels_predict, labels.long(), reduction='mean')

                self.reset_grad()
                c_loss.backward()
                # 备注：这里为了简便没有对优化器进行参数断点记录！！！
                self.g_optimizer.step()
                self.lr_scheduler.step(epoch=None)
                # scalr info on tensorboard
                if (i_iter + 1) % self.tb_step == 0:
                    self.writer.add_scalar(
                        'cross_entrophy_loss', c_loss.data, i_iter)
                    self.writer.add_scalar(
                        'learning_rate', self.g_optimizer.param_groups[0]['lr'], i_iter)
                # Sample images
                if (i_iter + 1) % self.sample_step == 0:
                    labels_sample = generate_label(
                        labels_predict, self.imsize)
                    save_image(denorm(labels_sample.data),
                               osp.join(self.sample_path, '{}_predict.png'.format(i_iter + 1)))
                # 每隔100次迭代输出一次信息
                if (i_iter + 1) % 100 == 0:
                    # 计算估计的训练时间
                    # 计算每100个iter耗费的时间
                    elapsed_time = time.time() - start_time1
                    # 估计每个iter消耗的时间
                    estimated_time_per_iter = elapsed_time / 100
                    # 估计每个batch消耗的时间
                    estimated_time_per_batch = estimated_time_per_iter * len(self.data_loader)
                    estimated_time_per_batch = str(datetime.timedelta(seconds=int(estimated_time_per_batch)))
                    # 估计总时间
                    estimated_total_time = estimated_time_per_iter * self.total_iters
                    # 秒转换为小时分钟秒
                    estimated_total_time = str(datetime.timedelta(seconds=int(estimated_total_time)))
                    # 重置开始时间
                    start_time1 = time.time()
                    # 记录实际的训练时间
                    usetime = time.time() - start_time2
                    usetime = str(datetime.timedelta(seconds=int(usetime)))
                    # 写入日志格式 - 模型名称 - 当前epoch/总epoch - 当前iter/总iter - 当前loss - 当前lr - 估计总时间 - 实际总时间 - Batch时间
                    root_logger.info('Model {} - Epoch {}/{} - Iteration {}/{} - Loss {} - Lr {} Etime - {} Utime - {} Btime {}'.format
                                     (self.arch, epoch + 1, self.epochs, i_iter, self.total_iters, c_loss.data,
                                      self.g_optimizer.param_groups[0]['lr'], estimated_total_time, usetime, estimated_time_per_batch))
                    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # print('{} - epoch={}/{} iter={} of {} completed, loss={}'
                    #       .format(timestamp, epoch, self.epochs, i_iter,self.total_iters, c_loss.data))

            # 每个epoch结束后进行一次验证
            miou = self.verifier.validation(self.G)
            # 记录miou到tensorboard
            self.writer.add_scalar('miou', miou, epoch)
            if miou > best_miou:
                best_miou = miou
                torch.save(self.G.state_dict(), osp.join(
                    self.model_save_path, '{}_{}_G.pth'.format(str(epoch), str(round(best_miou, 4)))))
        # 显式关闭进度条，良好习惯
        progress_bar.close()
    def build_model(self):
        resnet18Url="https://download.pytorch.org/models/resnet18-5c106cde.pth"
        resnet34Url="https://download.pytorch.org/models/resnet34-333f7ec4.pth"
        resnet50Url="https://download.pytorch.org/models/resnet50-19c8e357.pth"
        resnet101Url="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
        resnet152Url="https://download.pytorch.org/models/resnet152-b121ed2d.pth"
        if self.arch == "CE2P":
            self.G = get_model(self.arch, url=resnet101Url, pretrained=self.indicator).cuda()
        else:
            self.G = get_model(self.arch, pretrained=self.indicator).cuda()

        if self.parallel:
            self.G = nn.DataParallel(self.G)
        # Loss and optimizer
        self.g_optimizer = torch.optim.SGD(filter(
            lambda p: p.requires_grad, self.G.parameters()), self.g_lr, self.momentum, self.weight_decay)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(osp.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()

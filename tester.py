import os.path as osp
import torch
import timeit
import numpy as np
import datetime
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.utils import save_image
# from torchvision import transforms

from tqdm import tqdm
import logging
from networks import get_model
from utils import *
# from PIL import Image
import time
from metrics import SegMetric


# def make_dataset(dir):
#     images = []
#     assert osp.isdir(dir), '%s is not a valid directory' % dir

#     f = dir.split('/')[-1].split('_')[-1]
#     print(dir, len([name for name in os.listdir(dir)
#                     if osp.isfile(osp.join(dir, name))]))
#     for i in range(len([name for name in os.listdir(dir) if osp.isfile(osp.join(dir, name))])):
#         img = str(i) + '.jpg'
#         path = osp.join(dir, img)
#         images.append(path)

#     return images



class Tester(object):

    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        # self.imsize = config.imsize
        self.parallel = config.parallel
        self.classes = config.classes
        self.pretrained_model = config.pretrained_model  # int type

        self.model_save_path = config.model_save_path
        self.arch = config.arch
        # self.test_size = config.test_size

        self.build_model()

    def test(self):
        # Get current date and time
        now = datetime.datetime.now()
        # Format filename with current date and time
        filename = 'test{}_{}.log'.format(self.arch, now.strftime('%Y%m%d_%H-%M-%S'))
        # Configure logging
        logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # 创建一个StreamHandler，用于将日志输出到终端
        console_handler = logging.StreamHandler()
        # 设置日志等级
        console_handler.setLevel(logging.INFO)
        # 设置日志格式
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        # 获取根logger，并添加StreamHandler
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(console_handler)
        root_logger.info("Start testing...")
        # 记录开始测试时间
        time_meter = AverageMeter()
        # Model loading
        self.G.load_state_dict(torch.load(
            osp.join(self.model_save_path, self.arch, "{}_G.pth".format(self.pretrained_model))))
        self.G.eval()
        # batch_num = int(self.test_size / self.batch_size)
        metrics = SegMetric(n_classes=self.classes)
        metrics.reset()

        start_time = timeit.default_timer()
        for index, (images, labels) in enumerate(self.data_loader):
            if (index + 1) % 100 == 0:
                print('%d processd' % (index + 1))

            images = images.cuda()
            labels = labels.cuda()
            h, w = labels.size()[1], labels.size()[2]

            torch.cuda.synchronize()
            tic = time.perf_counter()

            with torch.no_grad():
                outputs = self.G(images)
                # Whether or not multi branch?
                if self.arch == 'CE2P' or 'FaceParseNet' in self.arch:
                    outputs = outputs[0][-1]

                outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
                gt = labels.cpu().numpy()
                elapsed_time = timeit.default_timer() - start_time
                # 将信息写入日志文件
                logging.info("Inference time (iter {0:5d}): {1:3.5f} fps".
                             format(index + 1, pred.shape[0] / elapsed_time))
                # print(
                #     "Inference time \
                #       (iter {0:5d}): {1:3.5f} fps".format(
                #         index + 1, pred.shape[0] / elapsed_time
                #     )
                # )
                metrics.update(gt, pred)

            torch.cuda.synchronize()
            time_meter.update(time.perf_counter() - tic)
        # elapsed_time = timeit.default_timer() - start_time
        # print("Inference time: {}fps".format(self.test_size / elapsed_time))
        print("Inference Time: {:.4f}s".format(
            time_meter.average() / images.size(0)))
        root_logger.info("Inference Time: {:.4f}s".format(
            time_meter.average() / images.size(0)))
        score = metrics.get_scores()[0]
        class_iou = metrics.get_scores()[1]

        for k, v in score.items():
            print(k, v)
            root_logger.info('{}: {}'.format(k, v))
        print("=========================================")
        facial_names = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                        'left_ear',
                        'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace', 'neck',
                        'cloth']
        for i in range(self.classes):
            print(facial_names[i] + "\t: {}".format(str(class_iou[i])))
            root_logger.info(facial_names[i] + "\t: {}".format(str(class_iou[i])))

    def build_model(self):
        self.G = get_model(self.arch, pretrained=False).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

import argparse


def str2bool(v):  # 定义一个函数，将字符串转换为布尔值
    return v.lower() in ('true')  # 如果字符串是'true'（不区分大小写），则返回True，否则返回False

def get_parameters():  # 定义一个函数，用于获取命令行参数
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象

    parser.add_argument('--imsize', type=int, default=512)  # 添加一个命令行参数，名为'imsize'，类型为整数，缺省值为512
    # 添加一个命令行参数，名为'arch'，类型为字符串，必须是列表中的一个值，此参数是必需的
    parser.add_argument(
        '--arch', type=str, choices=['UNet', 'DFANet', 'DANet', 'DABNet', 'CE2P', 'FaceParseNet18',
                                     'FaceParseNet34', "FaceParseNet50", "FaceParseNet101", "EHANet18", "EHAwithDDGCNet"], required=True)

    # 添加一些命令行参数，用于设置训练过程
    parser.add_argument('--epochs', type=int, default=200,
                        help='how many times to update the generator')  # 训练的轮数
    parser.add_argument('--pretrained_model', type=int, default= 0)  # 预训练模型的编号
    parser.add_argument('--batch_size', type=int, default=16)  # 批处理大小
    parser.add_argument('--num_workers', type=int, default=4)  # 工作线程的数量
    parser.add_argument('--g_lr', type=float, default=0.001)  # 生成器的学习率
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # 权重衰减
    parser.add_argument('--momentum', type=float, default=0.9)  # 动量
    parser.add_argument('--classes', type=int, default=19)  # 类别的数量

    # 添加一些命令行参数，用于设置测试过程
    parser.add_argument('--model_name', type=str, default='model.pth')  # 模型的名称

    # 添加一些其他的命令行参数
    parser.add_argument('--train', type=str2bool, default=True)  # 是否进行训练
    parser.add_argument('--parallel', type=str2bool, default=False)  # 是否并行处理

    #添加参数用于选择数据集
    parser.add_argument('--dataset', type=str, default='celeba')  # 数据集名称

    # 添加一些命令行参数，用于设置路径
    parser.add_argument('--img_path', type=str,
                        default='./Data_preprocessing/train_img')  # 训练图像的路径
    parser.add_argument('--label_path', type=str,
                        default='./Data_preprocessing/train_label')  # 训练标签的路径
    parser.add_argument('--model_save_path', type=str, default='./models')  # 模型保存的路径
    parser.add_argument('--sample_path', type=str, default='./samples')  # 样本的路径
    parser.add_argument('--val_img_path', type=str,
                        default='./Data_preprocessing/val_img')  # 验证图像的路径
    parser.add_argument('--val_label_path', type=str,
                        default='./Data_preprocessing/val_label')  # 验证标签的路径
    parser.add_argument('--test_image_path', type=str,
                        default='./Data_preprocessing/test_img')  # 测试图像的路径
    parser.add_argument('--test_label_path', type=str,
                        default='./Data_preprocessing/test_label')  # 测试标签的路径
    parser.add_argument('--test_color_label_path', type=str,
                        default='./test_color_visualize')  # 测试颜色标签的路径

    # 添加一些命令行参数，用于设置步长
    parser.add_argument('--sample_step', type=int, default=200)  # 样本步长
    parser.add_argument('--tb_step', type=int, default=100)  # TensorBoard步长

    return parser.parse_args()  # 解析命令行参数，并返回结果

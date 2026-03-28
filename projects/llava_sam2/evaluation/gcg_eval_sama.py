import argparse                             # 用于解析命令行参数
import math                                 # 提供数学函数，比如后面用到ceil
import os                                   # 提供操作系统接口，例如路径、文件操作
import torch                                # PyTorch的核心库，用于张量运算、深度学习
import tqdm                                 # 一个用于显示进度条的库
from pycocotools import mask as mask_utils  # COCO格式的掩码处理库，支持RLE等编码
import numpy as np                          # 数值计算库
from transformers import (                  # HuggingFace提供的transformers库，
    AutoModel,                              #   可自动加载预训练模型
    AutoModelForCausalLM,                   #   用于加载causal language model
    AutoTokenizer,                          #   自动加载相应模型的tokenizer
    BitsAndBytesConfig,                     #   量化相关的配置（此处未实际使用）
    CLIPImageProcessor,                     #   CLIP图像预处理器（此处未实际使用）
    CLIPVisionModel,                        #   CLIP视觉模型（此处未实际使用）
    GenerationConfig                        #   文本生成的配置
)

from utils import _init_dist_pytorch, get_dist_info, collect_results_cpu  # 自定义工具函数，分布式初始化、获取rank、汇总结果等
from PIL import Image                                                     # 图像处理库，读写图像
import re                                                                 # 正则表达式库
import json                                                               # JSON读写库

def parse_args():
    parser = argparse.ArgumentParser(description='GCG')               # 创建命令行参数解析器，描述“GCG”任务
    parser.add_argument('--model_path', help='hf model path.')        # 模型路径参数
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')                                       # 数据集的split名称，默认“val”
    parser.add_argument(
        '--save_dir',
        default='./gcg_pred/',
        help='save path')                                             # 推理结果的保存目录，默认'./gcg_pred/'
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')                                          # 分布式启动器类型，可选'none','pytorch','slurm','mpi'
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)  # 本地进程的rank（分布式相关）
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)               # 若没有LOCAL_RANK，则写入环境变量
    return args

# 指定一个图像文件夹路径，放置需要推理的图像
IMAGE_FOLDER = '/Sa2VA/reproduce/Sa2VA/data/glamm_data/images/grandf/val_test'

class GCGInferenceDataset:
    """一个简单的数据集类，用于读取指定文件夹下的图像，并构造推理需要的问题文本。"""
    def __init__(self,
                 image_folder,
                 save_dir=None,
                 ):
        self.image_folder = image_folder                 # 保存图像文件夹路径
        self.images = os.listdir(image_folder)           # 列出该文件夹内所有文件（图像名称列表）

        if save_dir is not None:
            # 如果提供了save_dir，就过滤掉已经生成过结果的图像文件，避免重复推理
            self.save_dir = save_dir
            exsits_files = os.listdir(self.save_dir)             # 已经输出过的文件
            exsits_files = [_file[:-5] for _file in exsits_files]# 去掉.json后缀
            _images = []
            for i, item in enumerate(self.images):
                if item[:-4] not in exsits_files:               # 如果该图像尚未被处理过
                    _images.append(item)
            self.images = _images

    def __len__(self):
        return len(self.images)                       # 返回数据集大小（即剩余需要推理的图像个数）

    def get_questions(self):
        """返回一个固定问题，用于请求对图像做简要描述并插入分割掩码"""
        question = "Could you please give me a brief description of the image? Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."
        return question

    def __getitem__(self, index):
        data_dict = {}
        questions = self.get_questions()                  # 获取固定问题字符串
        image_file = self.images[index]                   # 当前索引对应的图像文件名
        data_dict['image_file'] = image_file              # 保存图像文件名到data_dict

        image_file = os.path.join(self.image_folder, image_file)  # 拼接完整路径
        image = Image.open(image_file).convert('RGB')             # 用PIL读取图像并转为RGB模式

        data_dict['image'] = image                                # 将图像对象存入
        data_dict['text'] = "<image>\n" + questions               # 给模型的输入文本，包含<image>占位符 + 问题

        data_dict['img_id'] = image_file                          # 用图像路径充当ID
        return data_dict

def main():
    args = parse_args()  # 解析命令行参数

    # 根据launcher参数决定是否做分布式初始化
    if args.launcher != 'none':
        _init_dist_pytorch('nccl')             # 初始化pytorch分布式模式，后端为NCCL
        rank, world_size = get_dist_info()     # 获取当前进程的rank、总进程数
        torch.cuda.set_device(rank)            # 将当前进程绑定到对应的GPU
    else:
        rank = 0
        world_size = 1

    # 加载模型：从给定的args.model_path加载，指定数据类型等
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # 若结果存放目录不存在，则创建
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # 构建推理数据集对象
    dataset = GCGInferenceDataset(
        image_folder=IMAGE_FOLDER,
        save_dir=args.save_dir,
    )

    results = []                                # 用于收集所有推理的文本结果
    n_samples = len(dataset)                    # 数据集中需要推理的图像数量
    per_rank_samples = math.ceil(n_samples / world_size) + 1  # 每个进程要处理的图像数量（多加1是保险）
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))  # 当前进程实际需要处理的索引范围

    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]                                       # 从数据集中取出一个条目
        prediction = {'img_id': data_batch['img_id'], 'image_file': data_batch['image_file']}  # 记录当前图像ID和文件名
        del data_batch['img_id'], data_batch['image_file']             # 删除不再需要的键，后续会喂给模型predict_forward

        w, h = data_batch['image'].size                                 # 记录图像宽高，用于后面padding掩码尺寸

        # 调用模型的predict_forward函数进行推理
        pred_dict = model.predict_forward(**data_batch, tokenizer=tokenizer, prediction_only=False)
        # 判断predict_forward是否返回了分割掩码
        if 'prediction_masks' not in pred_dict.keys() or pred_dict['prediction_masks'] is None or len(pred_dict['prediction_masks']) == 0:
            print("No SEG !!!")
            # 若没有分割掩码就给一个空tensor
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
        else:
            # 将返回的分割掩码统一转换为tensor，并在dim=0方向堆叠
            raw_masks = []
            for mask in pred_dict['prediction_masks']:
                # 如果是numpy数组就先转tensor
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                raw_masks.append(mask)
            prediction['prediction_masks'] = torch.stack(raw_masks, dim=0)[:, 0]   # 只取第0通道: shape=(n,h,w)

        # 将当前图像的文本与掩码信息落盘
        process_and_save_output(
            args.save_dir,
            prediction['image_file'],
            pred_dict['prediction'],
            prediction['prediction_masks']
        )
        # 收集文本预测结果
        results.append(pred_dict['prediction'])

    # 若是多进程推理，聚合所有进程的结果
    results = collect_results_cpu(results, len(dataset), tmpdir='./gcg_eval_tmp')

def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    """后处理并将结果保存为json，包括文本和RLE格式的掩码。"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 去除多余字符，如<s>，换行等
    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    # 如果生成文本里有"ASSISTANT: "，只保留最后一次出现后面的文字
    text_output = text_output.split("ASSISTANT: ")[-1]

    # 用正则去除任何形如<...>的标签，保留干净的文本
    cleaned_str = re.sub(r'<.*?>', '', text_output)

    # 提取形如<p>段落</p>的短语
    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # 移除 [SEG] 标记
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # 去掉多余空格，以及首尾的引号
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # 将模型预测的掩码张量转为CPU，做RLE编码
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # 构建输出字典，包括图像ID、文本caption、提取到的phrases以及分割掩码
    result_dict = {
        "image_id": image_name[:-4],  # 去掉后缀
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    # 将结果写到对应json文件
    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    将分割掩码编码成未压缩RLE格式，便于使用pycocotools进行后续序列化或解码。
    tensor形状为 (b, h, w)。
    """
    # 首先将(h, w)的维度交换，以便flatten成连续的向量
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # 计算相邻像素的变化位置（diff非0的地方）
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # 将变化位置编码成长度信息
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    """将未压缩RLE通过pycocotools的frPyObjects进一步转成标准COCO RLE（counts二进制字符串）"""
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    # pycocotools生成的counts是bytes，需要解码成UTF-8字符串方可JSON化
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle

if __name__ == '__main__':
    main()

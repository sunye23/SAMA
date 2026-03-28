#!/usr/bin/env python3
import os
import re
import sys
import json
import copy
import time
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse
from google import genai
from google.genai import types
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

# 自定义工具函数（请确保 utils.py 中有相应实现）
from utils import load_mevis_json, filter_empty_instances, parse_json_for_objects, generate_region_descriptions

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 默认处理参数
MAX_FRAMES = 16
TARGET_SIZE = 384
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']

# ------------------ 工具函数 ------------------
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_list_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []

def append_to_file(file_path, item):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{item}\n")


def process_frame(frame_idx, video_data, file_name, mask_dict, image_root):
    """
    对单帧进行处理：
    1. 根据 frame 索引加载图像；
    2. 遍历 video_data['objects'] 中的每个物体，获取其所有 anno_ids 对应的 mask，
       选择第一个有效的 mask后，根据 mask 信息绘制目标边框与标识文本；
    3. 返回带框的 PIL 图像对象。
    """
    # 获取当前帧名称，并构造图像路径
    frame_name = video_data["frames"][frame_idx]
    img_path = os.path.join(image_root, file_name, f"{frame_name}.jpg")
    image = utils.read_image(img_path, format="RGB")
    original_shape = image.shape[:2]  # (height, width)

    # 使用 PIL 打开图像，并根据 TARGET_SIZE 进行缩放
    pil_img = Image.fromarray(image)
    width, height = pil_img.size
    if max(width, height) > TARGET_SIZE:
        scale_factor = TARGET_SIZE / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    else:
        new_width, new_height = width, height
    pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(pil_img)

    # 获取对象字典（注意：此处 video_data["objects"] 为字典）
    objects = video_data.get('objects', {})
    padding_top = 0
    padding_applied = False

    # 遍历每个物体
    for obj_id, obj_value in objects.items():
        masks = []
        # 遍历该物体所有的 anno_ids
        for anno in obj_value.get('anno_ids', []):
            mask_info = mask_dict.get(str(anno), {})
            if mask_info and frame_idx < len(mask_info):
                segm = mask_info[frame_idx]
                if segm:
                    # 保持原始的 COCO-style RLE 格式，不做 np.array 转换
                    masks.append(segm)
        if not masks:
            continue
        # 若有多个 mask，可考虑合并；这里直接取第一个有效的 mask
        merged_segm = masks[0]
        segm = merged_segm

        # 构造临时 annotation，用于后续 transform 与 box 计算
        obj_annotation = {
            "id": int(obj_id),
            "segmentation": segm,
            "category_id": 0,
            "bbox": [0, 0, 0, 0],
            "bbox_mode": BoxMode.XYXY_ABS
        }
        annos = utils.transform_instance_annotations(obj_annotation, [], original_shape)
        instances = utils.annotations_to_instances([annos], original_shape, mask_format="bitmask")
        if not instances.has("gt_masks"):
            continue
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances = filter_empty_instances(instances)
        if not (instances.has("gt_boxes") and len(instances.gt_boxes) > 0):
            logger.warning(f"No valid ground truth boxes for obj {obj_id} in frame {frame_idx}")
            continue

        # 使用第一个有效的边框
        box = instances.gt_boxes.tensor[0]
        if box.numel() != 4:
            logger.warning(f"Invalid box dimensions for obj {obj_id} in frame {frame_idx}")
            continue
        x1, y1, x2, y2 = box.tolist()
        scale_x = new_width / width
        scale_y = new_height / height
        x1, y1 = x1 * scale_x, y1 * scale_y
        x2, y2 = x2 * scale_x, y2 * scale_y

        # 检查文本绘制空间，不足时增加 padding
        try:
            font = ImageFont.truetype("arial.ttf", 25)
        except IOError:
            font = ImageFont.load_default()
        text = f'obj{obj_id}'
        text_bbox = font.getbbox(text)
        text_height = text_bbox[3] - text_bbox[1]
        if y1 - text_height < 0 and not padding_applied:
            padding_top = text_height + 5
            new_img = Image.new("RGB", (new_width, new_height + padding_top), (255, 255, 255))
            new_img.paste(pil_img, (0, padding_top))
            pil_img = new_img
            draw = ImageDraw.Draw(pil_img)
            padding_applied = True
        if padding_applied:
            y1 += padding_top
            y2 += padding_top

        color = COLORS[int(obj_id) % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - text_height - 2), text, fill=color, font=font)

    return pil_img


def process_video(video_name, video_data, mask_dict, image_root, system_prompt_file, output_dir):
    """
    对单个视频进行处理：
    1. 加载系统 prompt 并根据视频中对象信息生成 Gemini 问题；
    2. 对视频帧进行遍历处理，并保存带标注的可视化结果；
    3. 返回生成的 prompt（若视频帧数不足则返回 None）。
    """
    logger.info(f"开始处理视频：{video_name}")
    try:
        with open(system_prompt_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except Exception as e:
        logger.error(f"加载系统 prompt 失败：{e}")
        return None

    # 解析视频中的对象与描述
    objects, common_descriptions = parse_json_for_objects(video_data)
    question_num = min(len(objects) * 2, 8)
    descriptions = generate_region_descriptions(objects, common_descriptions)
    prompt = system_prompt.replace('<PLACEHOLDER>', descriptions)
    prompt = prompt.replace('<QUESTION_NUMBER>', str(question_num))
    questions = []
    questions.append(prompt)
    frames = sorted(video_data.get("frames", []))
    video_length = len(frames)
    if video_length < 2:
        logger.info(f"视频 {video_name} 帧数不足，跳过")
        return None

    if video_length > MAX_FRAMES:
        step = video_length / MAX_FRAMES
        selected_idx = [int(i * step) for i in range(MAX_FRAMES)]
    else:
        selected_idx = list(range(0, video_length, 2))

    logger.info(f"视频 {video_name} 总帧数：{video_length}，选取 {len(selected_idx)} 帧进行处理")
    annotated_images = []
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # 遍历选中的帧，保存带标注图像
    for frame_idx in tqdm(selected_idx, desc="处理帧", unit="frame"):
        try:
            annotated_img = process_frame(frame_idx, video_data, video_name, mask_dict, image_root)
            out_path = os.path.join(video_output_dir, f'frame_{frame_idx}_with_boxes.jpg')
            annotated_img.save(out_path)
            questions.append(annotated_img)
        except Exception as e:
            logger.error(f"处理视频 {video_name} 帧 {frame_idx} 时出错：{e}")
            continue

    return questions

def annotate_video_with_gemini(system_prompt):
    # genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=system_prompt)
        return response
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None

################################################################
# 3. 解析 Gemini 返回的文本，转成 conversation 格式
################################################################
def parse_gemini_response(response_text):
    """
    使用正则分别提取 description 和 conversation，兼容两个 JSON 独立返回的格式。
    """
    try:
        description = None
        conversation = None

        # 提取 description 块
        desc_match = re.search(r'"description"\s*:\s*(\{[\s\S]*?\})', response_text)
        if desc_match:
            desc_json = desc_match.group(1)
            description = json.loads(desc_json)

        # 提取 conversation 块（可能包含多个 QA 对）
        conv_match = re.search(r'"conversation"\s*:\s*(\[[\s\S]*?\])', response_text)
        if conv_match:
            conv_json = conv_match.group(1)
            raw_conv = json.loads(conv_json)
            conversation = [{"human": item["Question"], "gpt": item["Answer"]} for item in raw_conv]

        return description, conversation
    except Exception as e:
        print(f"[ERROR] Parsing failed: {e}")
        return None, None

# ------------------ 主函数 ------------------
def main():
    parser = argparse.ArgumentParser(description="Gemini 视频标注脚本")
    parser.add_argument('--meta_json_path', type=str, required=True, help='标注 JSON 文件路径')
    parser.add_argument('--mask_json_path', type=str, required=True, help='mask JSON 文件路径')
    parser.add_argument('--image_root', type=str, required=True, help='图像根目录')
    parser.add_argument('--error_file', type=str, required=True, help='错误视频记录文件路径')
    parser.add_argument('--response_txt', type=str, required=True, help='Gemini 响应文本文件路径')
    parser.add_argument('--system_prompt_file', type=str, required=True, help='系统 prompt 文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='可视化结果保存目录')
    parser.add_argument('--current_api_key', type=str, default=None, help='Gemini API KEY (可选，若不传则从环境变量读取)')
    args = parser.parse_args()
    # 设置全局 API KEY
    current_api_key = args.current_api_key if args.current_api_key else os.getenv('CURRENT_API_KEY')

    # 加载 JSON 数据和 mask 信息
    try:
        mevis_data = load_json_file(args.meta_json_path)
        mask_dict = load_json_file(args.mask_json_path)
        error_list = load_list_from_file(args.error_file)
    except Exception as e:
        logger.error(f"加载 JSON 文件出错：{e}")
        sys.exit(1)

    video_names = list(mevis_data.get("videos", {}).keys())
    logger.info(f"数据集中视频数：{len(video_names)}")

    # 遍历所有视频
    for video_name in video_names:
        if video_name in error_list:
            continue

        try:
            video_data = copy.deepcopy(mevis_data["videos"][video_name])
            if 'conversation' in video_data:
                continue
            prompt = process_video(video_name, video_data, mask_dict, args.image_root, args.system_prompt_file, args.output_dir)
            if prompt is None:
                logger.info(f"视频 {video_name} 跳过（帧数不足或其他问题）")
                continue

            # 调用 Gemini API 获取响应
            gemini_response = annotate_video_with_gemini(prompt)

            if gemini_response is None or not gemini_response.text:
                logger.error(f"视频 {video_name} 未收到 Gemini 响应")
                append_to_file(args.error_file, video_name)
                continue

            # 记录 Gemini 原始响应
            with open(args.response_txt, 'a', encoding='utf-8') as f:
                f.write(f"FileName: {video_name}\n")
                f.write(gemini_response.text + "\n")

            description, conversation = parse_gemini_response(gemini_response.text)
            print(gemini_response.text)
            if description is None or conversation is None:
                append_to_file(args.error_file, video_name)
                continue

            # 更新视频数据
            mevis_data["videos"][video_name]["description"] = description
            mevis_data["videos"][video_name]["conversation"] = conversation
            logger.info(f"视频 {video_name} 处理成功")
            save_json_file(mevis_data, args.meta_json_path)
            logger.info(f"最终的 annotated JSON 文件已保存至 {args.meta_json_path}")
        except Exception as e:
            logger.error(f"视频 {video_name} 处理过程中出错：{e}")
            append_to_file(args.error_file, video_name)
            continue

    # 保存最终的 annotated JSON 文件
    try:
        save_json_file(mevis_data, args.meta_json_path)
        logger.info(f"最终的 annotated JSON 文件已保存至 {args.meta_json_path}")
    except Exception as e:
        logger.error(f"保存最终 JSON 文件出错：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

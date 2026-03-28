import numpy as np
from collections import deque

class ConversationManager:
    def __init__(self, max_length):
        # 使用 deque 并设置最大长度，超出长度时会自动移除最早元素
        self.queue = deque(maxlen=max_length)

    def add_conversation(self, question, answer):
        """
        添加一组问答到队列中。
        如果队列已满，则自动移除最早的问答对。
        """
        item = {
            'from': 'human',
            'question': question,
            'from': 'gpt',
            'answer': answer
        }
        # 注意，这里将问题和回答作为一个完整的元组或者一个包含两个键的字典项加入队列
        self.queue.append({'question': question, 'answer': answer})

    def merge_conversations(self):
        merged = ' '.join(
            f"USER: {item['question']} ASSISTANT: {item['answer']}\n"
            for idx, item in enumerate(self.queue, start=1)
        )
        return merged

def uniform_sample(total_len, sample_num):
    intervals = np.linspace(start=0, stop=total_len, num=sample_num + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs

def get_sparse_indices(total_frame_num, num_frames_sparse):
    if total_frame_num > num_frames_sparse:       # video is long, uniformly sample frames
        frame_idxs = uniform_sample(total_frame_num, num_frames_sparse)
        return sorted(frame_idxs)
    else:
        num_repeat = num_frames_sparse // total_frame_num
        num_sample = num_frames_sparse % total_frame_num
        frame_idxs = list(range(total_frame_num)) * num_repeat + uniform_sample(total_frame_num, num_sample)
        return sorted(frame_idxs)


def get_dense_indices(num_frames_temporal, num_frames_dense):
    intervals = np.linspace(start=0, stop=num_frames_temporal - 1, num=num_frames_dense + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
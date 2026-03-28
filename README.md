# SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models. [NeurIPS 2025]

<a href="https://arxiv.org/abs/2505.18812"><img src="https://img.shields.io/badge/arXiv-2505.18812-b31b1b.svg" alt="arXiv"></a>

🔥 Code for the SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models. 


## :rocket: Updates 
* **[2026/3/27]** The code will be released before April. Thank you for your patience.
* **[2025/9/21]** SAMA is accepted to **NeurIPS 2025**🔥! See you in San Diego!😉
## Citation
**If you find SAMA useful for your work, please kindly cite using the BibTeX 🙏🙏🙏:**
```bibtex
@inproceedings{sun2025sama,
  title={SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models},
  author={Sun, Ye and Zhang, Hao and Ding, Henghui and Zhang, Tiehua and Ma, Xingjun and Jiang, Yu-Gang},
  booktitle={NeurIPS},
  year={2025}
}
```

## Contents 
- [Introduction](#introduction)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Training Data preparation](#training-data-preparation)
- [Training](#training)
- [Evaluation & Benchmark](#evaluation--benchmark)
- [Experiments](#experiments)
- [Data annotation](#data-annotation)
- [Acknowledgments](#acknowledgments)

## Introduction
Achieving fine-grained spatio-temporal understanding in videos remains a major challenge for current Video Large Multimodal Models (Video LMMs). Addressing this challenge requires mastering two core capabilities: video referring understanding, which captures the semantics of video regions, and video grounding, which segments object regions based on natural language descriptions.
However, most existing approaches tackle these tasks in isolation, limiting progress toward unified, referentially grounded video interaction. We identify a key bottleneck in the lack of high-quality, unified video instruction data and a comprehensive benchmark for evaluating referentially grounded video chat.
To address these challenges, we contribute in three core aspects: dataset, model, and benchmark.
First, we introduce SAMA-239K, a large-scale dataset comprising 15K videos specifically curated to enable joint learning of video referring understanding, grounding, and multi-turn video chat.
Second, we propose the SAMA model, which incorporates a versatile spatio-temporal context aggregator and a Segment Anything Model to jointly enhance fine-grained video comprehension and precise grounding capabilities.
Finally, we establish SAMA-Bench, a meticulously designed benchmark consisting of 5,067 questions from 522 videos, to comprehensively evaluate the integrated capabilities of Video LMMs in multi-turn, spatio-temporal referring understanding and grounded dialogue.
Extensive experiments and benchmarking results show that SAMA not only achieves strong performance on SAMA-Bench but also sets a new state-of-the-art on general grounding benchmarks, while maintaining highly competitive performance on standard visual understanding benchmarks.

<p align="center">
  <img src="/resources/sama_teaser.png" width="1000"/>
</p>

## Installation
<details open>
<summary>Installation</summary>

1. Please install the python and pytorch first:
```bash
> conda create -n vlm python=3.10
> conda activate vlm
> conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 cuda -c pytorch  -c "nvidia/label/cuda-12.1.0" -c "nvidia/label/cuda-12.1.1"
```

2. Install mmcv:
```bash
> pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
```

3. Install other dependencies:
```bash
> pip install -r requirements.txt
```
</details>

## Model Weights
We provide the following models:
| Model Name |                       HF Link                        |
|:----------:|:----------------------------------------------------:|
|  SAMA-1B  | [🤗 link](https://huggingface.co/Sunye2311/SAMA_1B) |
|  SAMA-4B  | [🤗 link](https://huggingface.co/Sunye2311/SAMA_4B) |
|  SAMA-8B  | [🤗 link](https://huggingface.co/Sunye2311/SAMA_8B) |

## Training Data preparation

<details open>
<summary>Data Preparation</summary>

1. Please first download the Sa2VA training datasets and place them in the `data` directory. The download link is [here](https://huggingface.co/datasets/Dense-World/Sa2VA-Training).

2. To support the training of SAMA239K, please first download the [LVVIS dataset](https://github.com/haochenheheda/LVVIS) and the [RefYoutube-VOS dataset](https://youtube-vos.org/dataset/rvos/) into the **sama239k_data** folder.

3. Create symbolic links in sama239k_data folder for the mevis dataset and the sav_train dataset (sam_v_full). These two datasets can be obtained from the Sa2VA training data.

4. For the [VidSTG dataset](https://github.com/Guaranteer/VidSTG-Dataset), we have performed frame extraction. Please download this dataset first and conduct frame extraction using our provided `/tools/vidstg_process.py`. The data organization format of VidSTG supported by the code is similar to the following: 
```
VidSTG_VIDEO_DIR/
├── video1.mp4
├── video2.mp4
└── video3.mp4

VidSTG_JSON_DIR/
├── video1.json
├── video2.json
└── video3.json
```

5. Download our json files [here](https://huggingface.co/datasets/Sunye2311/SAMA_Dataset) and put them into sama239k_data folder.

The final data structure should be like:
```
data/
├── sama239k_data
|   ├── mevis
|   |   └── train
|   ├── lvvis
|   |   └── train
|   ├── ref_youtube_vos
|   |   └── train
|   ├── sav_train
|   |   └── sav_000
|   |   └── .....
|   ├── VidSTG
|   |   └── train
|   |       └── 2399224635
|   |           └── frame_0.jpg
|   |           └── frame_4.jpg
|   |           └── .....
|   ├── sama239k_train_final.json          # sama 239k json file
|   ├── mevis_train_mask_dict.json         # reorganized mask annotation files
|   ├── lvvis_train_mask_dict.json
|   ├── ref_youtube_train_mask_dict.pkl
|   ├── SAV_mask_dict_train.json
|   ├── VidSTG_mask_dict_train_updated.json
├── video_datas
|   ├── revos
|   ├── mevis
|   └── davis17
|   └── chat_univi
|   └── sam_v_full # [!important] please download this from sam-2 directly.
|   └── Ref-SAV.json
├── ref_seg
|   ├── refclef
|   ├── refcoco
|   ├── refcoco+
|   ├── refcocog
|   ├── 
├── glamm_data
|   ├── images
|   ├── annotations
├── osprey-724k
|   ├── Osprey-724K
|   ├── coco
├── llava_data
|   ├── llava_images
|   ├── LLaVA-Instruct-150K
|   ├── LLaVA-Pretrain

```
</details>

## Training
To complete the training of SAMA, please first prepare the Sa2VA model weights and ensure that the dataset paths, model paths, and other configurations in the config file are set correctly.
```bash
> bash scripts/train/run_train_1b.sh
```
```bash
> bash scripts/train/run_train_4b.sh
```
```bash
> bash scripts/train/run_train_8b.sh
```
After training is complete, use the script below to convert and obtain the final model weights.
```bash
> bash scripts/model_convert_st.sh
```

## Evaluation & Benchmark
1. Image Segmentation: Example evaluation scripts for datasets such as Refcoco/+/g.
```bash
> bash scripts/inference/eval_refcoco.sh
```

2. Video Segmentation: Example evaluation scripts for datasets such as MeViS/Ref-Davis/Ref-youtube/ReVOS.
```bash
> bash scripts/inference/eval_mevis.sh
> bash tools/eval_video_seg/scripts/eval_mevis_metrics.sh
```

3. SAMA-Bench-G Evaluation:

Download our SAMA-Bench json files [here](https://huggingface.co/datasets/Sunye2311/SAMA_Dataset). The SAMA-Bench test set is drawn from the validation splits of four open-source video datasets: MeViS, LVVIS, Ref-YouTube-VOS, and VidSTG. Among them, the videos in the VidSTG validation split must also be processed into extracted frames using the provided `/tools/vidstg_process.py`. We have reorganized or re-annotated the mask annotation files of these datasets. In addition, due to the long evaluation time, we split the SAMA-Bench JSON file into multiple subsets, allowing you to utilize multiple nodes simultaneously to accelerate inference. Running inference on SAMA-Bench-g requires at least an A100-80G GPU. Since the videos in VidSTG are relatively long, the total inference time on 8 * A100-80G is approximately 4 hours. During evaluation, we primarily use the first-frame prompt of the query object as input to the model.

The expected data directory structure is as follows:
```
sama_bench
├── mevis                        # Video Image files
|   └── val_u
|           └── JPEGImages
├── lvvis
|   └── val
|           └── JPEGImages
├── ref_youtube_vos
|   └── valid
|           └── JPEGImages
├── VidSTG
|    └── val
|        └── 2400171624
|            └── frame_0.jpg
|            └── frame_4.jpg
|            └── .....
├── Mevis_mask_dict_val.json      # mask annotation files
├── LVVIS_mask_dict_val.json
├── RefYoutube_mask_dict_val.json
├── VidSTG_mask_dict_val_updated.json
├── lvvis_0.json                  # SAMA-Bench JSON files
├── lvvis_1.json
├── mevis.json
├── ref_youtube_vos_0.json
├── ref_youtube_vos_1.json
├── VidSTG_0.json
├── VidSTG_1.json
├── VidSTG_2.json
├── VidSTG_3.json
├── VidSTG_4.json
├── VidSTG_5.json
├── VidSTG_6.json
├── VidSTG_7.json
```
Example evaluation scripts for sama-bench-g:
```bash
> bash scripts/inference/eval_sama_bench_g.sh
> python scripts/inference/compute_sama_bench_g_final.py
```

4. SAMA-Bench-C Evaluation:
```bash
> bash scripts/inference/eval_sama_bench_c.sh
> python scripts/inference/compute_sama_bench_c_final.py
```

5. General Benchmark Evaluation: For the evaluation of general benchmarks such as MME and VideoMME, we primarily use [VLMEvalKit](https://github.com/open-compass/vlmevalkit).

## Experiments

## Data annotation

## Acknowledgments


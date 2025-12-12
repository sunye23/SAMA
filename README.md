# SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models. [NeruIPS 2025]

<a href="https://arxiv.org/abs/2505.18812"><img src="https://img.shields.io/badge/arXiv-2505.18812-b31b1b.svg" alt="arXiv"></a>

🔥 Code for the SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models. 


## :rocket: Updates 
* **[2025/12/12]** We are in the process of preparing the data. Please wait a moment.
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
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Training Data preparation](#training-data-preparation)
- [Training](#training)
- [Evaluation & Benchmark](#evaluation--benchmark)
- [Acknowledgments](#acknowledgments)

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

## Training Data preparation

<details open>
<summary>Data Preparation</summary>

1. Please first download the Sa2VA training datasets and place them in the `data` directory. The download link is [here](https://huggingface.co/datasets/Dense-World/Sa2VA-Training).

2. To support the training of SAMA239K, please first download the [LVVIS dataset](https://github.com/haochenheheda/LVVIS) and the [RefYoutube-VOS dataset](https://youtube-vos.org/dataset/rvos/) into the **sama239k_data** folder.

3. Create symbolic links in sama239k_data folder for the mevis dataset and the sav_train dataset (sam_v_full). These two datasets can be obtained from the Sa2VA training data.

4. For the [VidSTG dataset](https://github.com/Guaranteer/VidSTG-Dataset), we have performed frame extraction. Please download this dataset first and conduct frame extraction using our provided `/tools/vidstg_process.py`.

5. Download our json files here and put them into sama239k_data folder.

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

## Evaluation & Benchmark

## Acknowledgments


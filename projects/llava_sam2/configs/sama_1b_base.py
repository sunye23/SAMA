from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.llava_sam2.models.internvl_sama import InternVL_Slowfast

from projects.llava_sam2.models import VideoLLaVASAMModel, SAM2TrainRunner, VideoLLaVASAMModel_zero3
from projects.llava_sam2.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset
from projects.llava_sam2.datasets import VideoChatUniViDataset
from projects.llava_sam2.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset, OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset
from projects.llava_sam2.datasets import LLaVADataset
from projects.llava_sam2.datasets import ReferSegmDataset
from projects.llava_sam2.datasets import VideoRegionDescriptionDataset
from projects.llava_sam2.datasets import VideoRegionConvDataset

from projects.llava_sam2.models.preprocess.image_resize import DirectResize
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = '/path/to/Sa2VA-1B'
pretrained_pth = '/path/to/Sa2VA-1B'

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

#append parameters
NUM_TEMPORAL_TOKENS = 32

NUM_SPATIAL_TOKENS = 32
WINDOW_SIZE = 512
WINDOW_STRIDE = 512

DENSE_LEN=5
SPARSE_LEN=32
MAX_CONV_LEN=6
MAX_FRAMES=32

SELECTED_DATASETS = None
NUM_TEMPORAL_TOKENS_LLaVA = 32
MAX_CONV_LEN_LLaVA=1
QA_PER_ITEM=4
DESCRIPTION_PER_ITEM=3
max_steps = 10000

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        freeze_visual_encoder=True,
        llm_lora=None,
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    # loss_sample_points=False,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = '/path/to/data/'
VIDEO_DATA_ROOT = DATA_ROOT + 'video_datas/'

############### video res
data_root_revos = VIDEO_DATA_ROOT + 'revos/'
video_revos_image_folder = data_root_revos
video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
video_revos_mask_file = data_root_revos + 'mask_dict.json'

data_root_mevis = VIDEO_DATA_ROOT + 'mevis/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_mask_file = data_root_mevis + 'mask_dict.json'

data_root_refytvos = VIDEO_DATA_ROOT + 'rvos/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = data_root_refytvos + 'meta_expressions/train/meta_expressions.json'
video_refytvos_mask_file = data_root_refytvos + 'ref_youtube_train_mask_dict.pkl'
sav_path = None
vidstg_path = None
video_revos_dataset = dict(
    type=VideoReVOSDataset,
    image_folder=video_revos_image_folder,
    expression_file=video_revos_expression_file,
    mask_file=video_revos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

video_mevis_dataset = dict(
    type=VideoMeVISDataset,
    image_folder=video_mevis_image_folder,
    expression_file=video_mevis_expression_file,
    mask_file=video_mevis_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    mask_file=video_refytvos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

################### Video chat
data_root_video_chatunivi = VIDEO_DATA_ROOT + 'chat_univi/'
video_chatunivi_image_folder = data_root_video_chatunivi + 'Activity_Videos/'
video_chatunivi_json_file = data_root_video_chatunivi+ 'video_chat.json'

video_qa_dataset = dict(
    type=VideoChatUniViDataset,
    image_folder=video_chatunivi_image_folder,
    json_file=video_chatunivi_json_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

################## image chat
LLAVA_ROOT = DATA_ROOT + 'llava_data/'
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='/path/to/llava_v1_5_mix665k.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    num_temporal_tokens=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_stride=WINDOW_STRIDE,
    max_conv_len=MAX_CONV_LEN_LLaVA,
    image_folder=LLAVA_ROOT + 'llava_images/',
    window_size=WINDOW_SIZE,
)

################## image res
RES_ROOT = DATA_ROOT + 'ref_seg/'
refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/path/to/coco/refcoco',
    data_prefix=dict(img_path='/path/to/coco/train2014'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_length=max_length,
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/path/to/coco/refcoco+',
    data_prefix=dict(img_path='/path/to/coco/train2014'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_length=max_length,
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/path/to/coco/refcocog',
    data_prefix=dict(img_path='/path/to/coco/train2014'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_length=max_length,
)

# image gcg datas
glamm_data_root = DATA_ROOT + 'glamm_data/'

refcocog_image_path = glamm_data_root + 'images/coco2014/train2014/'
refcocog_ann_file = glamm_data_root + 'annotations/RefCOCOg_GCG_train.json'

grandf_image_path = glamm_data_root + 'images/grandf/train/'
grandf_ann_file = glamm_data_root + 'annotations/GranDf_HA_GCG_train.json'

flickr_image_path = glamm_data_root + 'images/flickr30k/Flickr30K/'
flickr_ann_file = glamm_data_root + 'annotations/flickr_mergedGT_GCG_train.json'

psg_image_path = glamm_data_root + 'images/coco2017/'
psg_ann_file = glamm_data_root + 'annotations/OpenPsgGCG_train.json'

glamm_refcocog_dataset = dict(
    type=RefCOCOgGCGDataset,
    image_folder=refcocog_image_path,
    data_path=refcocog_ann_file,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

glamm_grandf_dataset = dict(
    type=GranDfGCGDataset,
    data_path=grandf_ann_file,
    image_folder=grandf_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=10,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

glamm_psg_dataset = dict(
    type=OpenPsgGCGDataset,
    data_path=psg_ann_file,
    image_folder=psg_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

glamm_flickr_dataset = dict(
    type=FlickrGCGDataset,
    data_path=flickr_ann_file,
    image_folder=flickr_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# sam2 data
data_sam2_folder = '/path/to/ref_sav/sam_v_full'
data_sam2_expression_file = '/path/to/ref_sav/Ref-SAV.json'

video_sam2_dataset = dict(
    type=VideoSAM2Dataset,
    sam2_folder=data_sam2_folder,
    expression_file=data_sam2_expression_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    select_number=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

# osprey
OSPREY_ROOT = DATA_ROOT + "osprey-724k/"
data_osprey_file = OSPREY_ROOT + 'Osprey-724K/osprey_conversation.json'
data_osprey_image_folders = [
    OSPREY_ROOT + 'coco/train2014/',
    OSPREY_ROOT + 'coco/val2014/',
    OSPREY_ROOT + 'coco/train2017/',
    OSPREY_ROOT + 'coco/val2017/',
]
#
image_osprey_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

data_osprey_detail_description_file = OSPREY_ROOT + 'Osprey-724K/osprey_detail_description.json'
image_osprey_description_dataset = dict(
    type=OspreyDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_detail_description_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)
data_osprey_short_file = OSPREY_ROOT + 'Osprey-724K/osprey_short_form.json'
image_osprey_short_dataset = dict(
    type=OspreyShortDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_short_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=0,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

data_osprey_part_file = OSPREY_ROOT + 'Osprey-724K/osprey_part_level.json'
image_osprey_part_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_part_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

data_osprey_positive_neg_file = OSPREY_ROOT + 'Osprey-724K/osprey_lvis_positive_negative.json'
image_osprey_positive_neg_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_positive_neg_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

data_video_region_descrpition_folders = '/datasets/datasets_sysy/llava_all_image_video'
data_video_region_descrpition_files = '/Sa2VA/reproduce/Sa2VA/video_region_meta/sama239k_train_final.json'

video_region_description_dataset = dict(
    type=VideoRegionDescriptionDataset,
    image_folder=data_video_region_descrpition_folders,
    data_path=data_video_region_descrpition_files,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=5,
    sampled_frames=3,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    description_per_item=DESCRIPTION_PER_ITEM,
    special_tokens=special_tokens,
    mevis_json='/datasets/datasets_sysy/llava_all_image_video/mevis_train_mask_dict.json',
    lvvis_json='/datasets/datasets_sysy/llava_all_image_video/lvvis_train_mask_dict.json',
    ref_youtube_pkl=video_refytvos_mask_file,
    sav_json='/Sa2VA/reproduce/Sa2VA/video_region_meta/mask_dict/Train/SAV_mask_dict_train.json',
    vidstg_json='/Sa2VA/reproduce/Sa2VA/video_region_meta/mask_dict/Train/VidSTG_mask_dict_train_updated.json',
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
    selected_datasets=SELECTED_DATASETS,
)

data_video_region_conv_folders = '/datasets/datasets_sysy/llava_all_image_video'
data_video_region_conv_files = '/Sa2VA/reproduce/Sa2VA/video_region_meta/sama239k_train_final.json'

video_region_conv_dataset = dict(
    type=VideoRegionConvDataset,
    image_folder=data_video_region_conv_folders,
    data_path=data_video_region_conv_files,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=5,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    dense_len=DENSE_LEN,
    max_conv_len=MAX_CONV_LEN,
    qa_per_item=QA_PER_ITEM,
    mevis_json='/datasets/datasets_sysy/llava_all_image_video/mevis_train_mask_dict.json',
    lvvis_json='/datasets/datasets_sysy/llava_all_image_video/lvvis_train_mask_dict.json',
    ref_youtube_pkl=video_refytvos_mask_file,
    sav_json='/Sa2VA/reproduce/Sa2VA/video_region_meta/mask_dict/Train/SAV_mask_dict_train.json',
    vidstg_json='/Sa2VA/reproduce/Sa2VA/video_region_meta/mask_dict/Train/VidSTG_mask_dict_train_updated.json',
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
    selected_datasets=SELECTED_DATASETS,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # sem seg
        # semantic_seg_ade20k_dataset,
        # ref seg
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # # # image qa
        llava_vqa_dataset,
        # # # # video res
        video_mevis_dataset, video_revos_dataset, video_refytvos_dataset,
        # # # # # video chatas
        video_qa_dataset,
        # # # # # # # # # sam2 pesudo
        video_sam2_dataset,
        # # # # # # # # # gcg data
        glamm_psg_dataset,
        glamm_grandf_dataset,
        glamm_flickr_dataset,
        glamm_refcocog_dataset,
        # # # # # # # # # # visual prompt
        image_osprey_dataset, image_osprey_description_dataset,
        image_osprey_part_dataset, image_osprey_short_dataset,
        image_osprey_positive_neg_dataset,
        video_region_conv_dataset,
        video_region_description_dataset,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=True,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

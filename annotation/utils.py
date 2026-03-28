import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from PIL import Image
import json
import logging
import numpy as np

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm

logger = logging.getLogger(__name__)
LVVIS_CATEGORIES = ["accordion", "action camera", "aerosol can", "air conditioner", "air fryer", "airplane", "alarm clock", "alcohol", "alcohol lamp", "alligator", "almond", "ambulance", "amplifier", "anklet", "antelope", "antenna", "apple", "applesauce", "apricot", "apron", "armband", "armchair", "armoire", "armor", "army tank", "artichoke", "ashtray", "asparagus", "atomizer", "automatic washer", "avocado", "award", "awning", "ax", "baboon", "baby buggy", "backhoe", "backpack", "badger", "bagpipe", "baguet", "bait", "balance scale", "balances", "ball", "ballet skirt", "balloon", "ballpoint pen", "bamboo", "bamboo shoots", "banana", "band aid", "bandage", "banjo", "banner", "barbell", "barcode", "barrel", "barrette", "barrow", "baseball", "baseball bat", "baseball cap", "baseball glove", "basket", "basketball", "basketball backboard", "bass guitar", "bass horn", "bassoon", "bat (animal)", "bath mat", "bath towel", "bathrobe", "bathtub", "battery", "beach towel", "beachball", "beaker", "bean curd", "beanbag", "beanie", "bear", "bed", "bedpan", "beef (food)", "beeper", "beer bottle", "beer can", "beetle", "bell", "bell pepper", "belt", "belt buckle", "beluga whale", "bench", "beret", "bib", "bible", "bicycle", "bicycle pump", "billards", "billboard", "binder", "binder clip", "binoculars", "bird", "birdbath", "birdcage", "birdfeeder", "birdhouse", "birthday cake", "birthday card", "blackboard", "blanket", "blazer", "blender", "blimp", "blouse", "blue whale", "blueberry", "board eraser", "boat", "bobby pin", "body thermometer", "boiled egg", "bolo tie", "bolt", "bongos", "book", "bookcase", "bookend", "bookmark", "boot", "bottle", "bottle cap", "bottle opener", "bouquet", "bow (decorative ribbons)", "bow (weapon)", "bow-tie", "bowl", "bowler hat", "bowling ball", "box", "boxing glove", "bracelet", "brass plaque", "brassiere", "bread", "bread-bin", "briefcase", "broccoli", "broom", "bubble gum", "bucket", "building blocks", "bulldog", "bulldozer", "bulletin board", "bulletproof vest", "bullhorn", "bun", "bunk bed", "buoy", "burette", "bus (vehicle)", "butter", "butterfly", "button", "cab (taxi)", "cabbage", "cabinet", "cable car", "cage", "cake", "calculator", "calendar", "camcorder", "camel", "camera", "camera lens", "camera tripod", "camper (vehicle)", "can", "can opener", "candle", "candle holder", "candy bar", "candy cane", "canister", "canoe", "cantaloup", "canteen", "canvas bag", "car (automobile)", "car battery", "car jack", "car odometer", "carabiner", "card", "card game", "cardigan", "cargo ship", "carnation", "carpet", "carrot", "cart", "carton", "casserole", "cassette", "cassette player", "cat", "cauliflower", "CD", "cd player", "cell phone charger", "cello", "cellular telephone", "centipede", "centrifuge", "certificate", "chain mail", "chair", "chaise longue", "chalice", "chalk", "chandelier", "chap", "charger", "checkbook", "cheese curls", "cheetah", "chef hat", "cheongsam", "cherry", "chessboard", "chestnut", "chewing gum", "chicken (animal)", "chickpea", "chili (vegetable)", "chime", "chinaware", "chisel", "chocolate bar", "choker", "chopping board", "chopstick", "christmas tree", "cicada", "cigar box", "cigarette", "clam", "clarinet", "clasp", "claw hammer", "cleat (for securing rope)", "clippers (for plants)", "cloak", "clock", "clock tower", "clothes hamper", "clothespin", "coaster", "coat", "coat hanger", "coatrack", "cockatoo", "cockroach", "coconut", "cod", "coffee maker", "coffee mug", "coffeepot", "coin", "combination lock", "comic book", "compass", "compass (drawing tool)", "computer box", "computer keyboard", "condiment", "cone", "convertible (automobile)", "cooler (for food)", "cork (bottle plug)", "corkscrew", "cornet", "correction fluid", "correction tape", "cotton swab", "cougar", "cover", "coverall", "cow", "cowboy hat", "crab (animal)", "cranberry", "crane", "crate", "crawfish", "crayon", "cream pitcher", "crib", "crisp (potato chip)", "crock pot", "crossbar", "crowbar", "crucifix", "cruise ship", "crutch", "cucumber", "cup", "cupcake", "curling iron", "curtain", "cushion", "cuttlefish", "cymbal", "dagger", "dalmatian", "dartboard", "darts", "date (fruit)", "deadbolt", "deck chair", "deer", "defibrillator", "dehumidifier", "desk", "desk chair", "detergent", "diaper", "die", "dining table", "dirt bike", "discus", "dish", "dish antenna", "dishrag", "dishtowel", "dishwasher", "dispenser", "diving board", "dog", "dog collar", "doll", "dollar", "dolphin", "domestic ass", "doorknob", "doormat", "double-sided tape", "doughnut", "dove", "dragonfly", "dragonfruit", "drawer", "drawing board", "dress", "dress hat", "dresser", "drill", "drill bit", "drone", "dropper", "drum (musical instrument)", "drum set", "drumstick", "duck", "duct tape", "duffel bag", "dulcimer", "dumbbell", "dumpling", "dumpster", "durian", "dustpan", "e-cigarette", "e-reader", "eagle", "earmuffs", "earphone", "earplug", "earring", "easel", "eclair", "eel", "egg", "egg roll", "egg tart", "egg yolk", "eggbeater", "eggplant", "electric bicycle", "electric chair", "electric drill", "electric heater", "electric kettle", "elephant", "envelope", "eraser", "escargot", "excavator", "external hard drive", "eye liner", "eye shadow", "eyepatch", "face mask", "falcon", "fan", "faucet", "fedora", "ferret", "fig (fruit)", "file (tool)", "file folder", "fire alarm", "fire engine", "fire extinguisher", "fire truck", "firefly", "fireplace", "fireplug", "first-aid kit", "fish", "fishbowl", "fishing rod", "flag", "flagpole", "flamingo", "flash drive", "flashlight", "flea", "flip-flop (sandal)", "flower arrangement", "flowerpot", "flute", "fly", "folding chair", "folding knife", "football (american)", "football helmet", "footstool", "fork", "forklift", "fox", "fragrance", "freight car", "french horn", "fridge magnet", "fried chicken", "frisbee", "frog", "fruit juice", "frying pan", "fume hood", "funnel", "game console", "gameboard", "gamepad", "gaming chairs", "garbage truck", "garlic", "gas pipe", "gas stove", "gasmask", "gazelle", "gemstone", "generator", "giant panda", "gift wrap", "giraffe", "glass (drink container)", "glider", "globe", "glockenspiel", "glove", "glue", "go-kart", "goat", "goggles", "goldfish", "golf ball", "golf club", "golfcart", "goose", "gorilla", "gourd", "grape", "grapefruit", "grasshopper", "grater", "gravestone", "green onion", "green plants (potted plants)", "griddle", "grill", "guava", "guitar", "gun", "hair curler", "hair dryer", "hairbrush", "hairnet", "hairpin", "halter top", "ham", "hamburger", "hammer", "hammock", "hamper", "hamster", "hand grips strengthener", "hand towel", "handbag", "handcart", "handcuff", "handkerchief", "handle", "handsaw", "hang glider", "hard drive", "hardback book", "harmonium", "hat", "hatbox", "headband", "headboard", "headlamp", "headlight", "headscarf", "headset", "heart", "hedgehog", "helicopter", "helmet", "heron", "high heels", "high jump standards", "highchair", "hinge", "hippopotamus", "hockey stick", "hog", "honey", "hoodie", "hookah", "horse", "horse buggy", "hose", "hot dog", "hot-air balloon", "hotplate", "hourglass", "houseboat", "hovercraft", "humidifier", "hummingbird", "hurdle", "ice pack", "ice skate", "icecream", "igniter", "incense burner", "inflatable bed", "infusion pump", "iron (for clothing)", "ironing board", "jackal", "jacket", "jackfruit", "jaguar", "jar", "javelin", "jean", "jeep", "jellyfish", "jersey", "jet plane", "joystick", "jump rope", "kangaroo", "kayak", "keg", "kennel", "kettle", "kettlebell", "key", "keycard", "kimono", "kitchen paper", "kitchen sink", "kite", "kiwi fruit", "knee pad", "knife", "knob", "koala", "lab coat", "ladder", "ladybug", "lamb-chop", "lamp", "lamppost", "lampshade", "lantern", "lanyard", "laptop computer", "lasagna", "latch", "lawn mower", "leather shoes", "legging (clothing)", "lego", "lemon", "leopard cat", "lettuce", "level (tools)", "license plate", "life buoy", "life jacket", "lightbulb", "lighter", "lighthouse", "lightning rod", "lion", "lip balm", "lipstick", "liquor", "lizard", "loader", "lobster", "locker", "log", "lollipop", "long jump pit", "lychee", "machine gun", "magazine", "magic cube", "maglev", "magnet", "magpie", "mailbox (at home)", "mallard", "mammoth", "manatee", "mandolin", "mango", "manhole", "manual (instruction book)", "map", "marble", "marker", "marten", "martini", "mashed potato", "masher", "mask", "massage chair", "mast", "mat (gym equipment)", "match", "matchbox", "mealworms", "measuring stick", "meatball", "mechanical pencil", "medal", "megaphone", "melon", "memo pad", "microphone", "microscope", "microwave oven", "milestone", "milk", "milk can", "milkshake", "minivan", "mint candy", "mirror", "mitten", "mixer (kitchen tool)", "mole", "monitor (computer equipment)", "monkey", "moose", "mop", "mosquito", "motor vehicle", "motorcycle", "mouse (computer equipment)", "mousepad", "mug", "mushroom", "music box", "music stand", "music stool", "nail polish", "nailfile", "napkin", "nebulizer", "necklace", "necktie", "nectarine", "needle", "nest", "nightingale", "nightshirt", "nightstand", "noodle", "nosebag (for animals)", "noseband (for animals)", "notebook", "notepad", "nut", "nutcracker", "oar", "oboe", "octopus (animal)", "octopus (food)", "oil lamp", "oil tanker", "okra", "onion", "orange (fruit)", "organ", "ostrich", "otoscope", "otter", "oven", "owl", "oxygen concentrator", "oyster", "pad", "pad (electronic product)", "paddle", "padlock", "paint brush", "paintbrush", "painting", "palette", "pan (for cooking)", "papaya", "paper bag", "paper clip", "parachute", "parasail (sports)", "parchment", "parka", "parrot", "passenger ship", "passion fruit", "pasta strainer", "pastry", "peach", "peacock", "pear", "peeler (tool for fruit and vegetables)", "pegboard", "pelican", "pen", "pencil", "pencil box", "pencil sharpener", "pendulum", "penguin", "pennant", "persimmon", "person", "petri dish", "phonograph record", "piano", "pickaxe", "pickup truck", "picnic basket", "picture", "pigeon", "piggy bank", "pill", "pillow", "pin (non jewelry)", "pineapple", "pinecone", "ping-pong ball", "pinwheel", "pipe", "pirate flag", "pistol", "pizza", "place mat", "plastic bag", "plate", "platypus", "playpen", "pliers", "plow (farm equipment)", "plume", "pocket watch", "poker", "poker chip", "polar bear", "pole", "police car", "polo shirt", "pomegranate", "pool cue", "pool table", "popcorn", "popsicle", "postcard", "poster", "pot", "potato", "potholder", "pouch", "power bank", "power drill", "power saw", "praying mantis", "pressure cooker", "printer", "projector", "propeller", "protective suit", "protractor", "prune", "pudding", "puffer (fish)", "puffin", "pug-dog", "pumpkin", "puncher", "puppet", "puzzle", "quesadilla", "quiche", "quilt", "quince", "rabbit", "raccoon", "race car", "radar", "radiator", "radio receiver", "raft", "rag doll", "raincoat", "raisins", "ramen", "rat", "razor", "razorblade", "rearview mirror", "record player", "reflector", "refrigerator", "relay baton", "remote control", "remote control car", "rhinoceros", "rice cooker", "rickshaw", "rifle", "ring", "road map", "roadblock", "roast duck", "robe", "rocket", "roller skate", "rollerblade", "rolling pin", "router (computer equipment)", "rowboat", "ruler", "runner (carpet)", "safety hammer", "safety pin", "sail", "sailboat", "salad plate", "salami", "salmon (fish)", "salmon (food)", "sandal (type of shoe)", "sandbag", "sandpaper", "sandwich", "sardine", "satchel", "saucepan", "sausage", "saxophone", "scallops", "scanner", "scarecrow", "scarf", "school bus", "scissors", "scoreboard", "scorpions", "screw", "screwdriver", "scrubbing brush", "sculpture", "SD card", "sea urchin", "seagull", "seahorse", "seal", "seaplane", "seashell", "seaweed", "sedan", "selfie stick", "sewing machine", "shampoo", "shark", "sharpie", "shaver (electric)", "shawl", "shears", "sheep", "shelf", "shepherd dog", "shield", "shirt", "shoe", "shoehorn", "shoeshine", "shopping cart", "short pants", "shot glass", "shot put", "shoulder bag", "shovel", "shower cap", "shower curtain", "shower head", "shredder (for paper)", "shrimp", "side table", "signboard", "sink", "skateboard", "ski", "ski parka", "ski pole", "skirt", "skunk", "sled", "sleeping bag", "slide", "slipper (footwear)", "smartwatch", "smoothie", "snake", "snow leopard", "snowboard", "snowman", "snowmobile", "soap", "soccer ball", "sock", "socket", "sofa", "sofa bed", "sombrero", "soundbar", "soupspoon", "soya milk", "space shuttle", "sparrow", "spatula", "speaker (stero equipment)", "spear", "spectacles", "speed bump", "sphygmomanometer", "spice rack", "spider", "spinach", "sponge", "spoon", "sportswear", "spotlight", "spring rolls", "squash", "squid (food)", "squirrel", "stapler (stapling machine)", "starfish", "starfruit", "starting blocks", "steak (food)", "steak knife", "steamroller", "steel drum", "steering wheel", "stepladder", "stereo (sound system)", "stethoscope", "sticker", "stirring rod", "stool", "stop sign", "strap", "straw (for drinking)", "strawberry", "street sign", "streetlight", "string cheese", "stuffed animal", "submarine", "subway", "subwoofer", "sugar bowl", "sugarcane (plant)", "suit (clothing)", "suitcase", "sunflower", "sunglasses", "surfboard", "surveillance cameras", "sushi", "suspenders", "swallow", "swan", "sweater", "sweatshirt", "sweet potato", "swim cap", "swim ring", "swimming goggles", "swimsuit", "swing", "sword", "synthesizer", "syringe", "table", "table lamp", "table-tennis table", "tablecloth", "tachometer", "tag", "taillight", "tambourine", "tangerine", "tank top (clothing)", "tape (sticky cloth or paper)", "tape measure", "tapestry", "target", "tarp", "tassel", "tea bag", "teakettle", "teapot", "telephone", "telephone pole", "telephoto lens", "television set", "tennis ball", "tennis racket", "tent", "tequila", "termites", "test tube", "test tube holder", "thermometer", "thermostat", "thimble", "thumbtack", "tiger", "tights (clothing)", "timer", "tinfoil", "tissue paper", "toast (food)", "toaster", "tobacco pipe", "toilet", "toilet tissue", "tomato", "tongs", "toolbox", "toothbrush", "toothpaste", "toothpick", "tow truck", "towel", "towel rack", "toy", "tractor (farm equipment)", "traffic light", "trailer truck", "train (railroad vehicle)", "trampoline", "trash can", "travel pillow", "tray", "trench coat", "triangle (musical instrument)", "tricycle", "tripod", "trombone", "trophy", "trousers", "trowel", "truck", "trumpet", "trunk", "tuba", "tuna", "turban", "turnip", "turtle", "tweezers", "typewriter", "umbrella", "unicycle", "urinal", "vacuum cleaner", "vase", "vending machine", "vent", "vest", "vinegar", "viola", "violin", "virtual reality headset", "visor", "vodka", "volleyball", "vulture", "waffle", "wagon wheel", "waist pack", "walkie talkie", "walking stick", "walkman", "wall socket", "wallet", "walnut", "walrus", "warthog", "washbasin", "wasp", "watch", "water bottle", "water cooler", "water gun", "water heater", "water jug", "water ski", "water temperature gauge", "watering can", "watermelon", "webcam", "weightlifting belt", "welding torch", "wet suit", "wheel", "wheelchair", "whipped cream", "whistle", "white sugar", "whiteboard", "wig", "wind chime", "windmill", "windshield wiper", "wine bottle", "wine bucket", "wineglass", "wireless chargers", "wok", "wolf", "wood plane", "wooden spoon", "woodpecker", "wreath", "wrench", "wristband", "wristlet", "writing brush", "xylophone", "xylophone mallets", "yo-yo", "yoga mat", "zebra", "zucchini"]

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())
    r.append(instances.gt_classes != -1)

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    return instances

def load_mevis_json(image_root_=None, json_file_=None):
    if image_root_ is None:
        image_root = '/remote-home/sunye/video_project/video_dataset/mevis/train'
    else:
        image_root = image_root_
    if json_file_ is None:
        json_file = '/remote-home/sunye/video_project/video_dataset/mevis/train/meta_expressions_cleaned.json'
    else:
        json_file = json_file_
    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0

    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    print('number of video in the datasets:{}'.format(len(videos)))
    metas = []
    if image_root.split('/')[-1] == 'train':
        mask_json = os.path.join(image_root, 'mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            mask_dict = json.load(fp)

        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 2:
                continue
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                meta['length'] = vid_len
                metas.append(meta)
    else:
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = -1
                meta['anno_id'] = -1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                meta['length'] = vid_len
                metas.append(meta)

    dataset_dicts = []
    for vid_dict in tqdm(metas):
        record = {}
        record["file_names"] = [os.path.join(image_root, 'JPEGImages', vid_dict['video'], vid_dict["frames"][i]+ '.jpg') for i in range(vid_dict["length"])]
        record["length"] = vid_dict["length"]
        video_name, exp, anno_ids, obj_ids, category, exp_id = \
            vid_dict['video'], vid_dict['exp'], vid_dict['anno_id'], vid_dict['obj_id'], vid_dict['category'],  vid_dict['exp_id']
        exp = [" ".join(sentence.lower().split()) for sentence in exp]
        if "eval_idx" in vid_dict:
            record["eval_idx"] = vid_dict["eval_idx"]

        video_objs = []
        if image_root.split('/')[-1] == 'train':
            for frame_idx in range(record["length"]):
                frame_objs = []
                for x, obj_id in zip(anno_ids, obj_ids):
                    obj = {}
                    segm = mask_dict[x][frame_idx]
                    if not segm:
                        num_instances_without_valid_segmentation += 1
                        continue
                    num_instances_valid_segmentation += 1
                    bbox = [0, 0, 0, 0]
                    obj["id"] = obj_id
                    obj["segmentation"] = segm
                    obj["category_id"] = category
                    obj["bbox"] = bbox
                    obj["bbox_mode"] = BoxMode.XYXY_ABS
                    frame_objs.append(obj)
                video_objs.append(frame_objs)
        record["annotations"] = video_objs
        record["sentence"] = exp
        record["exp_id"] = exp_id
        record["video_name"] = video_name
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Total {} instance and Filtered out {} instances without valid segmentation. ".format(
                num_instances_valid_segmentation, num_instances_without_valid_segmentation
            )
        )

    return dataset_dicts

#
# def parse_json_for_objects(dataset_dict):
#     """
#     Parse the dataset_dict from the JSON to generate objects and common descriptions.
#
#     Args:
#     dataset_dict (dict): JSON content for a specific video.
#
#     Returns:
#     tuple: A sorted list of objects and a list of common descriptions.
#     """
#     objects = []
#     common_descriptions = []
#
#     # A mapping for object colors (for demonstration)
#     colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']
#
#     # Iterate through each expression in the dataset
#     for key, value in dataset_dict['expressions'].items():
#         obj_ids = value['obj_id']
#         exp_descriptions = value['exp']
#
#         # Process the sentences into a single string
#         sentence_raw = [" ".join(sentence.lower().split()).rstrip(".") for sentence in exp_descriptions]
#         if len(sentence_raw) > 5:
#             sentence_raw = sentence_raw[:5]
#         sentence_raw_str = ', '.join(sentence_raw)  # Join sentences with a comma
#
#         if len(obj_ids) == 1:
#             # Single object descriptions
#             obj_id = obj_ids[0]
#             objects.append((obj_id, colors[obj_id], sentence_raw_str))
#         else:
#             # Multiple objects descriptions (common)
#             common_descriptions.append((obj_ids, sentence_raw_str))
#
#     # Sort objects by obj_id in ascending order
#     objects = sorted(objects, key=lambda x: x[0])
#
#     return objects, common_descriptions
def parse_json_for_objects(dataset_dict):
    """
    Parse the dataset_dict from the JSON to generate objects and common descriptions.

    Args:
        dataset_dict (dict): JSON content for a specific video.

    Returns:
        tuple: A sorted list of objects and a list of common descriptions.
    """
    objects = []
    common_descriptions = []
    # 定义颜色映射
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']

    # 如果数据中存在 "objects" 键，则使用它
    if "objects" in dataset_dict:
        for key, value in dataset_dict["objects"].items():
            # key 表示对象的 id（字符串形式），转换为整型
            obj_id = int(key)
            # 从对象数据中获取表达列表，默认空列表
            exp_descriptions = value.get("expressions", [])
            # 对表达句子进行处理：小写、去除多余空格和结尾句号
            sentence_raw = [" ".join(sentence.lower().split()).rstrip(".") for sentence in exp_descriptions]
            # 限制句子数（例如最多 5 条）
            if len(sentence_raw) > 3:
                sentence_raw = sentence_raw[:3]
            sentence_raw_str = ', '.join(sentence_raw)
            # 将单个对象描述保存到 objects 中
            objects.append((obj_id, colors[obj_id % len(colors)], sentence_raw_str))
    else:
        # 如果没有 "objects"，尝试按旧格式从 "expressions" 键中读取
        for key, value in dataset_dict.get('expressions', {}).items():
            obj_ids = value['obj_id']
            exp_descriptions = value['exp']
            sentence_raw = [" ".join(sentence.lower().split()).rstrip(".") for sentence in exp_descriptions]
            if len(sentence_raw) > 5:
                sentence_raw = sentence_raw[:5]
            sentence_raw_str = ', '.join(sentence_raw)
            if len(obj_ids) == 1:
                obj_id = obj_ids[0]
                objects.append((obj_id, colors[obj_id % len(colors)], sentence_raw_str))
            else:
                common_descriptions.append((obj_ids, sentence_raw_str))

    # 按对象 id 升序排序
    objects = sorted(objects, key=lambda x: x[0])
    return objects, common_descriptions


def parse_json_for_objects_refyoutube(dataset_dict):
    """
    Parse the dataset_dict from the JSON to generate objects and common descriptions.

    Args:
    dataset_dict (dict): JSON content for a specific video.

    Returns:
    tuple: A sorted list of objects and a list of common descriptions.
    """
    objects = []
    common_descriptions = []

    # A mapping for object colors (for demonstration)
    colors = ['pink', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime']

    # Iterate through each expression in the dataset
    for key, value in dataset_dict['expressions'].items():
        obj_ids = value['obj_id']
        exp_descriptions = value['exp']

        # Process the sentences into a single string
        sentence_raw = [" ".join(sentence.lower().split()).rstrip(".") for sentence in exp_descriptions]
        if len(sentence_raw) > 5:
            sentence_raw = sentence_raw[:5]
        sentence_raw_str = ', '.join(sentence_raw)  # Join sentences with a comma

        if len(obj_ids) == 1:
            # Single object descriptions
            obj_id = obj_ids[0]
            objects.append((obj_id, colors[int(obj_id)], sentence_raw_str))

    # Sort objects by obj_id in ascending order
    objects = sorted(objects, key=lambda x: x[0])

    return objects

# def parse_json_for_objects_refyoutube(dataset_dict):
#     """
#     从 JSON 中解析对象信息，生成对象列表和（如果有）共同描述。
#     这里假设 JSON 中 "expressions" 部分的 key 就表示对象 id，
#     每个表达项只有 "exp" 字段。
#
#     Args:
#         dataset_dict (dict): 单个视频的 JSON 内容。
#
#     Returns:
#         tuple: (objects, common_descriptions)
#             objects: 按对象 id 升序排序的列表，每个元素为 (obj_id, box_color, description)
#             common_descriptions: 此处为空列表（因为只处理单个对象描述）
#     """
#     objects = []
#     common_descriptions = []  # 目前不处理多对象的共同描述
#
#     # 定义颜色列表（循环使用）
#     colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']
#
#     # 遍历每个表达项，key 为对象 id（字符串形式）
#     for key, value in dataset_dict.get('expressions', {}).items():
#         try:
#             obj_id = int(key)
#         except ValueError:
#             continue  # 如果转换失败则跳过
#
#         exp_description = value.get('exp', "")
#         # 预处理表达文本：小写、去多余空格和结尾句号
#         sentence = " ".join(exp_description.lower().split()).rstrip(".")
#         objects.append((obj_id, colors[obj_id % len(colors)], sentence))
#
#     # 按对象 id 升序排序
#     objects = sorted(objects, key=lambda x: x[0])
#     return objects

# def parse_json_for_objects_refyoutube(dataset_dict):
#     """
#     解析对象结构，返回：
#     [
#         {
#             "object_id": "0",
#             "main_exp": "a dog is with its puppies on the cloth",
#             "all_exps": ["a dog is with its puppies on the cloth", "a dog with puppies all around"]
#         },
#         ...
#     ]
#     """
#     objects = []
#     for obj_id, obj_info in dataset_dict.get("objects", {}).items():
#         exps = obj_info.get("exp", [])
#         if not exps:
#             continue
#         objects.append({
#             "object_id": obj_id,
#             "main_exp": exps[0],  # 取第一个作为主描述
#             "all_exps": exps
#         })
#     return objects





def parse_json_for_objects_lvvis(dataset_dict):
    """
    Parse the dataset_dict from the JSON to generate objects and common descriptions.

    Args:
    dataset_dict (dict): JSON content for a specific video.

    Returns:
    tuple: A sorted list of objects and a list of common descriptions.
    """
    objects = []

    # A mapping for object colors (for demonstration)
    colors = [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
        'lime', 'pink', 'brown', 'beige', 'navy', 'teal', 'violet', 'gold',
        'silver', 'coral', 'salmon', 'indigo'
    ]

    # Iterate through each expression in the dataset
    for key, value in dataset_dict['instance'].items():
        objects.append((int(key), colors[int(key)], LVVIS_CATEGORIES[value['category']]))

    # Sort objects by obj_id in ascending order
    objects = sorted(objects, key=lambda x: x[0])

    return objects

def generate_region_descriptions(objects, common_descriptions):
    """
    Generate descriptions for multiple objects, each with a color and a description,
    as well as common descriptions that apply only when these objects are together.

    Args:
    objects (list of tuples): Each tuple contains (obj_id, box_color, description).
    common_descriptions (list of tuples): Each tuple contains (obj_ids, description)
                                          for shared characteristics between objects.

    Returns:
    str: Generated region descriptions.
    """
    result = "{\n"

    # Generate individual descriptions
    for obj_id, box_color, description in objects:
        result += f'  "<obj{obj_id}> ({box_color} box)": "{description}",\n'
    # if len(common_descriptions) > 0:
    #     result += '\n'
    #     # Generate common descriptions for combined regions
    #     for obj_ids, description in common_descriptions:
    #         obj_ids_str = ' '.join([f"<obj{obj_id}>" for obj_id in obj_ids])
    #         result += f'  "Multi-object expressions for {obj_ids_str}": "{description}",\n'

    # Remove the last comma and close the JSON-like structure
    result = result.rstrip(",\n") + "\n}"

    return result


def generate_region_descriptions_youtube(objects):
    """
    Generate descriptions for multiple objects, each with a color and a description,
    as well as common descriptions that apply only when these objects are together.

    Args:
    objects (list of tuples): Each tuple contains (obj_id, box_color, description).
    common_descriptions (list of tuples): Each tuple contains (obj_ids, description)
                                          for shared characteristics between objects.

    Returns:
    str: Generated region descriptions.
    """
    result = "{\n"

    # Generate individual descriptions
    for obj_id, box_color, description in objects:
        result += f'  "<obj{obj_id}> ({box_color} box)": "{description}",\n'

    # Remove the last comma and close the JSON-like structure
    result = result.rstrip(",\n") + "\n}"

    return result

# def generate_region_descriptions_youtube(objects):
#     """
#     根据对象列表生成区域描述文本，每个对象对应一个描述。
#     对象列表中每个元素为 (obj_id, box_color, description)。
#
#     Returns:
#         str: 生成的描述文本（类似 JSON 格式的字符串）。
#     """
#     result = "{\n"
#     for obj_id, box_color, description in objects:
#         result += f'  "<obj{obj_id}> ({box_color} box)": "{description}",\n'
#     # 移除最后的逗号，并闭合结构
#     result = result.rstrip(",\n") + "\n}"
#     return result

# def generate_region_descriptions_youtube(objects):
#     """
#     objects: list of dicts, each with object_id, main_exp
#     返回 Gemini-friendly 区域描述文本
#     """
#     result = "{\n"
#     color_list = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']
#     for i, obj in enumerate(objects):
#         obj_id = obj['object_id']
#         exp = obj['main_exp']
#         color = color_list[int(obj_id) % len(color_list)]
#         result += f'  "<obj{obj_id}> ({color} box)": "{exp}",\n'
#     result = result.rstrip(",\n") + "\n}"
#     return result

def generate_region_descriptions_lvvis(objects):
    result = "{\n"

    # Generate individual descriptions
    for obj_id, box_color, description in objects:
        result += f'  "<obj{obj_id}> ({box_color} box)": "{description}",\n'

    # Remove the last comma and close the JSON-like structure
    result = result.rstrip(",\n") + "\n}"

    return result


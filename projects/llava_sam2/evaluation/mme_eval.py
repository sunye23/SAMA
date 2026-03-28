# mme_inmemory_eval_and_run_calculation.py
import os
import argparse
from PIL import Image
import torch
import tqdm
import math
from transformers import AutoModel, AutoTokenizer
from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
import shutil


def collect_mme_data(mme_root):
    """
    Function to collect and organize MME Benchmark data into a list of (image_path, question, answer).

    Steps:
    1) Traverse through subfolders like 'artwork', 'code_reasoning', etc.
    2) Check if there are 'images/' and 'questions_answers_YN/' directories.
    3) For each txt file, find the corresponding image (e.g., '339.txt' matches '339.jpg').
    4) Return a list of tuples: (relative_img_path, question, gt_answer).
    """
    records = []
    subfolders = sorted(os.listdir(mme_root))
    possible_exts = ['.jpg', '.png', '.jpeg']

    for subf in subfolders:
        subf_path = os.path.join(mme_root, subf)
        if not os.path.isdir(subf_path):
            continue

        # Check for 'images/' and 'questions_answers_YN/' subfolders
        images_dir = os.path.join(subf_path, 'images')
        qa_dir = os.path.join(subf_path, 'questions_answers_YN')
        has_images_subdir = os.path.isdir(images_dir)
        has_qa_subdir = os.path.isdir(qa_dir)

        if not has_images_subdir:
            images_dir = subf_path  # If no 'images/', images and txt are in the same directory
        if not has_qa_subdir:
            qa_dir = subf_path  # If no 'questions_answers_YN/', txt and images are in the same directory

        # Find all .txt files
        all_txt_files = [f for f in os.listdir(qa_dir) if f.endswith('.txt')]
        for txt_filename in all_txt_files:
            base_name = os.path.splitext(txt_filename)[0]  # e.g., "339"
            txt_path = os.path.join(qa_dir, txt_filename)

            # Find the corresponding image
            matched_image_name = None
            for ext in possible_exts:
                candidate_img_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(candidate_img_path):
                    matched_image_name = base_name + ext
                    break

            if matched_image_name is None:
                print(f"[Warning] No matching image found for {txt_path}, skipping.")
                continue

            # Construct relative image path
            if images_dir == subf_path:
                rel_img_path = os.path.join(subf, matched_image_name)
            else:
                rel_img_path = os.path.join(subf, 'images', matched_image_name)

            # Read the txt file for each question-answer pair
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue
                    question, gt_answer = parts[0], parts[1]
                    records.append((rel_img_path, question, gt_answer))

    return records


def parse_args():
    parser = argparse.ArgumentParser(description='In-Memory MME Evaluation and Automatic Calculation')
    parser.add_argument('--mme-root', type=str, required=True,
                        help='Path to the MME_Benchmark directory.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to your HF model (or custom model) directory.')
    parser.add_argument('--output-file', type=str, default='mme_results',
                        help='Output file path for 4-column results.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch',
                        help='Job launcher type')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args



def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # Load the model
    print("Step 1) Loading the model...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Collect data
    print("Step 2) Collecting data from MME Benchmark...")
    data_list = collect_mme_data(args.mme_root)
    print("total len data_list is: {}".format(len(data_list)))
    # Split data per rank
    n_samples = len(data_list)
    per_rank_samples = math.ceil(n_samples / world_size)
    start_idx = per_rank_samples * rank
    end_idx = min(n_samples, per_rank_samples * (rank + 1))
    per_rank_ids = range(start_idx, end_idx)

    # Run inference
    print("Step 3) Running inference on each QA pair...")
    results = []
    for idx in tqdm.tqdm(per_rank_ids):
        (rel_img_path, question, gt_answer) = data_list[idx]
        full_img_path = os.path.join(args.mme_root, rel_img_path)
        image = Image.open(full_img_path).convert('RGB')
        input_question = question
        if '<image>' not in question:
            input_question = "<image>" + '\n' + question
        try:
            out_dict = model.predict_forward(
                image=image,
                text=input_question,
                tokenizer=tokenizer
            )
            raw_resp = out_dict['prediction'].replace('<|im_end|>', '').replace('.', '').replace("\n", "").strip()
        except Exception as e:
            print(f"[Error] model prediction failed on {rel_img_path}: {e}")
            raw_resp = "Error"

        raw_resp = raw_resp.replace('\n', ' ').replace('\r', ' ')
        raw_resp = ' '.join(raw_resp.split())

        result_line = f"{rel_img_path}\t{question}\t{gt_answer}\t{raw_resp}"
        results.append(result_line)

    # Ensure `tmpdir` exists for the results
    tmpdir = os.path.join(args.output_file,
                          'dist_test_temp_res_' + args.model_path.replace('/', '').replace('.', '') + '_mme')
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # Call collect_results_cpu to gather results from all ranks
    results = collect_results_cpu(results, len(data_list), tmpdir=tmpdir)

    # Ensure `output_dir` exists
    output_dir = args.output_file  # This is a folder path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results by category
    if get_rank() == 0:
        print("results length in rank 0: {}".format(len(results)))

        # # First save the overall results to mme_results.txt
        # output_file = "/Sa2VA/reproduce/Sa2VA/model_zoo/Sa2VA/Sa2VA-1B/mme_results/mme_results.txt"
        # with open(output_file, "w", encoding='utf-8') as f:
        #     # Write each result line separately as a string
        #     for line in results:
        #         f.write(line + '\n')

        results_by_category = {}

        for line in results:
            parts = line.split('\t')
            if len(parts) < 4:
                continue

            rel_img_path, question, gt_answer, pred_answer = parts

            category = rel_img_path.split('/')[0]
            images = rel_img_path.split('/')[-1]

            # Check if the category is being correctly identified
            if category not in results_by_category:
                results_by_category[category] = []

            # Add result to the corresponding category
            results_by_category[category].append(f"{images}\t{question}\t{gt_answer}\t{pred_answer}")

        # Debugging: Print out the categories to check if everything is being added correctly
        print("Results grouped by category:")
        total_counts = 0
        for category, category_results in results_by_category.items():
            print(f"Category: {category}, Count: {len(category_results)}")
            total_counts = total_counts + len(category_results)
        print("total_counts is : {}".format(total_counts))
        # Save each category's results to its respective file
        for category, category_results in results_by_category.items():
            category_file = os.path.join(output_dir, f"{category}.txt")
            with open(category_file, 'w', encoding='utf-8') as f:
                for line in category_results:
                    f.write(line + '\n')

        print(f"Results saved in the directory: {output_dir}")


if __name__ == '__main__':
    main()
import os
import random
from collections import defaultdict
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy

from common import modeling
from eval.safe import config as safe_config
from eval.safe import search_augmented_factuality_eval as safe


def get_contents(dp):
    selected_index = 0 if dp["messages"][0]["role"] == "user" else 1
    assert dp["messages"][selected_index]["role"] == "user"
    assert dp["messages"][selected_index+1]["role"] == "assistant"
    prompt = dp["messages"][selected_index]["content"]
    response = dp["messages"][selected_index+1]["content"]
    return prompt, response


def decide_if_true(rater_model, dp, cache, i, save_interval, cache_path):
    prompt, response = get_contents(dp)

    if prompt+response in cache:
        result = cache[prompt+response]
    else:
        result = safe.main(prompt, response, rater_model)
        cache[prompt+response] = result
        if i % save_interval == 0:
            json.dump(cache, open(cache_path, 'w'))
    return {"Supported": result["Supported"], "Irrelevant": result["Irrelevant"], "Not Supported": result["Not Supported"]}


def main(args):
    random.seed(2024)
    results = json.load(open(args.result_path, 'r'))
    cache_path = os.path.join(args.cache_dir, 'serper.json')
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, 'r'))
    else:
        cache = {}

    for_validating_data = defaultdict(list)
    for dataset, data in results.items():
        labeled_data = data['data']
        for dp in labeled_data:
            if dp['is_factual']:
                for_validating_data[dataset].append(dp)

    for k, v in for_validating_data.items():
        num_existing_data = 0
        sampling_list = deepcopy(v)
        cached_list = []
        if args.use_existing_cache:
            for dp in v:
                prompt, response = get_contents(dp)
                if prompt+response in cache:
                    cached_list.append(dp)
                    sampling_list.remove(dp)
                    num_existing_data += 1
        random.shuffle(sampling_list)
        sample_num = max(args.n - num_existing_data, 0)
        print(f"For {k}: Cached {num_existing_data} datapoints, sampling {sample_num} more from {len(sampling_list)} datapoints.")
        if len(sampling_list) <= sample_num:
            print(f"Warning: not enough factual question for dataset: {k}!!!")
            assert False
        for_validating_data[k] = cached_list + sampling_list[:sample_num]

    rater_model = modeling.Model(
        safe_config.model,
        temperature=safe_config.model_temp,
        max_tokens=safe_config.max_tokens,
    )

    out_file = defaultdict(list)

    for dataset, data in for_validating_data.items():
        print(f"Processing {len(data)} for {dataset}...")
        for i, dp in enumerate(tqdm(data)):
            dp['is_true'] = decide_if_true(rater_model, dp, cache, i, args.save_interval, cache_path)
            out_file[dataset].append(dp)
        json.dump(cache, open(cache_path, 'w'))

    with open(os.path.join(args.output_path, f'hallucination_{args.n}.json'), 'w') as f:
        json.dump(out_file, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--use_existing_cache", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
    
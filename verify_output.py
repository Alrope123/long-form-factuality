import argparse
import json
from collections import defaultdict
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns


def get_contents(dp):
    selected_index = 0 if dp["messages"][0]["role"] == "user" else 1
    assert dp["messages"][selected_index]["role"] == "user"
    assert dp["messages"][selected_index+1]["role"] == "assistant"
    prompt = dp["messages"][selected_index]["content"]
    response = dp["messages"][selected_index+1]["content"]
    return prompt, response


def main(args):
    data = json.load(open(args.json_path, 'r'))
    cache = json.load(open(args.cache_path, 'r'))

    cat_to_not_supported_distribution = defaultdict(list)
    cat_to_not_supported_distribution_graph = defaultdict(list)
    for category, cat_data in data.items():
        for dp in cat_data:
            dp['is_true']['hallucination_rate'] = dp['is_true']['Not Supported'] / (dp['is_true']['Supported'] + dp['is_true']['Irrelevant'] + dp['is_true']['Not Supported'])
            prompt, response = get_contents(dp)
            checked_statements = cache[prompt+response]['checked_statements']
            unsupported_statements = []
            for statement in checked_statements:
                if statement["annotation"] == "Not Supported":
                    unsupported_statements.append({
                        "Fact": statement['self_contained_atomic_fact'],
                        'Reasoning': statement['rate_data']
                    })
            cat_to_not_supported_distribution[category].append(dp)
            dp['is_true']['hallucinations'] = unsupported_statements
            cat_to_not_supported_distribution_graph[category].append(dp['is_true']['hallucination_rate'])

    json.dump(cat_to_not_supported_distribution, open(args.output_path, 'w'), indent=4)

    for category, rates in cat_to_not_supported_distribution_graph.items():
        plt.title(f"{category}_not_supported_rate")
        sns.histplot(rates, stat='probability')
        plt.xlabel("Portion of Unsupported Atomic Facts")
        plt.ylabel("Percentage")
        plt.savefig(f"output/{category}_not_supported_rate.png")
        plt.clf()

        print(f"Average for {category}: {sum(rates) / len(rates)}, percentage of below {args.bar}: {sum([r < args.bar for r in rates]) / len(rates)}")

    totals = []
    [totals.extend(rates) for category, rates in cat_to_not_supported_distribution_graph.items()]
    plt.title("total_not_supported_rate")
    sns.histplot(totals, stat='probability')
    plt.xlabel("Portion of Unsupported Atomic Facts")
    plt.ylabel("Percentage")
    plt.savefig(f"output/total_not_supported_rate.png")
    plt.clf()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--bar", type=int, default=0.1)

    args = parser.parse_args()
    main(args)
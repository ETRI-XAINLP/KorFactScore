
import os
import sys

import jsonlines
import pandas as pd
import json
from tqdm import tqdm

def read_jsonl_file(file_path):
    l_result = []

    with jsonlines.open(file_path) as reader:
        for line in reader:
            l_result.append(line)

    return l_result


def do_main(out_dir, input_file, af_file):
    # read an input file
    input_dic = {}
    with open(input_file) as f:
        for line in f:
            dp = json.loads(line)
            input_dic[dp["topic"]] = dp

    # read an atomic_fact file
    dic_afs = {}
    with jsonlines.open(af_file) as reader:
        for l in reader:
            topic = l["topic"]
            l_afs = l["atomic_facts"]
            atomic_facts ={}
            for sent, afs in l_afs:
                atomic_facts[sent] = afs
            dic_afs[topic] = atomic_facts

    # combine
    input_with_af = []
    for t in input_dic.keys():
        if t in dic_afs:
            atomic_facts = dic_afs[t]
            anno_list = []
            for text in atomic_facts.keys():
                model_afs = atomic_facts[text]
                model_af_list = []
                for af in model_afs:
                    model_af_list.append({"text": af})
                anno_list.append({"text": text, "is-relevant": True, "model-atomic-facts": model_af_list})
            input_dic[t]["annotations"] = anno_list
        else:
            input_dic[t]["annotations"] = []

    # generate output
    in_dir, in_file = os.path.split(input_file)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, in_file)
    with jsonlines.open(output_file, mode='w', flush=True) as writer:
        for t in input_dic.keys():
            out_dic = input_dic[t]
            writer.write(out_dic)


if __name__ == '__main__':
    out_dir = "/home/qa/kmh/workspace/pycharm-2023.1/KorFactScore/data/k_labeled"
    input_file = "/home/qa/kmh/workspace/pycharm-2023.1/KorFactScore/data/k_unlabeled/gen_bio-gpt_4-kr-all.jsonl"
    af_file = "/home/qa/kmh/workspace/pycharm-2023.1/KorFactScore/cache/full1/gpt_4-kr-all/2_af_gen_results.jsonl"
    #
    do_main(out_dir, input_file, af_file)

    print(f"\n\t##### [ END of Program ] #####\n\n")

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import LlamaTokenizer

# from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

class CLM(LM):
    def __init__(self, model_name_or_path, cache_file=None):
        self.model_name = model_name_or_path
        # self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cache_file = kwargs.get("cache_file", None)
        # model_dir = kwargs.get("model_dir", None)
        instance = cls(model_name_or_path=model_name_or_path, cache_file=cache_file)
        return instance
    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        # pad_token_id가 None일 때만 eos_token_id를 사용하도록 조건을 설정
        pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        true_token_id = self.tokenizer.encode('True')[-1]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        for curr_input_ids in input_ids:
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id = pad_token_id
            )
            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0], true_token_id

        return generations, scores, true_token_id


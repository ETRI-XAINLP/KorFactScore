import os
import json
import jsonlines
import kss
from rank_bm25 import BM25Okapi
import random

from factscore.atomic_facts import best_demos, sent_tokenize_kor
from factscore.openai_lm import OpenAIModel
from factscore.auto_modelfactory import ModelFactory


class InvalidGenerationFilter(object):
    def __init__(self, key_path, filter_demon_file, lm_cache_file, filter_cache_file, model_name="gpt-3.5-turbo-0125", temperature=1.0):
        self.model_name = model_name
        self.temperature = temperature
        self.openai_lm = ModelFactory.from_pretrained(model_name, cache_file=lm_cache_file, key_path=key_path, temperature=self.temperature)
        self.filter_cache_file = filter_cache_file
        #
        self.verbose = True

        #
        def read_filter_demons(filter_demon_file):
            with open(filter_demon_file, 'r') as f:
                filter_demons = json.load(f)
            fkeys = filter_demons.keys()
            if len(fkeys) == 0:
                return None, None
            else:
                filter_demons_pos = []
                filter_demons_neg = []
            #
            for k in fkeys:
                if filter_demons[k] == True:
                    filter_demons_pos.append(k)
                else:
                    filter_demons_neg.append(k)
            #
            return filter_demons_pos, filter_demons_neg

        # get demos for filtering "abstained sentences"
        self.filter_demons_pos, self.filter_demons_neg = read_filter_demons(filter_demon_file)
        # TODO(kmh): active filter demons retrieval for positive case???
        # make bm25 for negative filtering demons
        tokenized_corpus = [doc.split(" ") for doc in self.filter_demons_neg]
        self.bm25_fta_neg = BM25Okapi(tokenized_corpus)
        #
        self.prompt_pref_filter_abstrained = "주어진 문장이 인물에 대한 실질적인 내용을 포함하는지 판별해주세요. 다른 문서의 참조를 권하거나 답변을 기권(abstain)한 경우도 '불포함'입니다. 인물에 대한 정보를 다음에 올 문장에서 제시하겠다고 한 경우 '불포함'입니다. 답변은 'True' 또는 'False' 중 하나로만 선택해주세요.\n(입력) "

    def generate_prompt_for_filter_abstain(self, topic, sent):
        # add a positive sample
        t = f"** {topic} ** "
        pos_sent = random.choice(self.filter_demons_pos)
        prompt = self.prompt_pref_filter_abstrained[:] + t + pos_sent[:] + "\n(판별 결과)\nTrue\n\n"
        # add a negative sample
        k = 1
        neg_sent = best_demos(sent, self.bm25_fta_neg, self.filter_demons_neg, k)
        prompt = prompt + self.prompt_pref_filter_abstrained[:] + t + neg_sent[0][:] + "\n(판별 결과)\nFalse\n\n"
        # add a query sentences
        prompt = prompt + self.prompt_pref_filter_abstrained[:] + t + sent[:] + "\n(판별 결과)\n"

        return prompt

    def query_lm_for_filtering(self, prompt):
        output, _ = self.openai_lm.generate(prompt)
        output = output.lower()
        if "true" in output:
            return True
        elif "false" in output:
            return False
        elif "abstain" in output:
            return False
        elif "기권" in output:
            return False
        else:
            print(f"[ERROR] Output is not true or false: {output}")
        return None

    def filter_abstained_sentences(self, topic, sentences):
        # TODO: save intermediate results for request failures by LLMs
        verdicts = {}
        meaningful_sentences = []
        for s in sentences:
            prompt = self.generate_prompt_for_filter_abstain(topic, s)
            qlf = self.query_lm_for_filtering(prompt)
            if qlf:
                meaningful_sentences.append(s)
                verdict = "pass"
            else:
                verdict = "ABSTAINED"
            #
            if self.verbose: print(f"[{verdict}] {s}", flush=True)
            verdicts[s] = qlf

        return meaningful_sentences, verdicts

    def get_init_sentences(self, generation, mode="gpt-4"):
        assert isinstance(generation, str), "generation must be a string"
        #
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]

        # filtering for "gemini"'s results
        if mode == "gemini":
            new_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                if len(para) > 0:
                    if '**' == para[:2] and '**' == para[-2:]:      # 시작과 끝이 "**"로 마킹되어 있는 줄은 제외
                        continue
                    else:
                        if '* ' == para[:2]:
                            para = para[2:]
                        new_paragraphs.append(para)
            paragraphs = new_paragraphs
        #
        init_sents = []
        for para_idx, paragraph in enumerate(paragraphs):
            curr_sentences = sent_tokenize_kor(paragraph)
            init_sents += curr_sentences
        #
        return init_sents

    def determine_fully_filtered(self, init_sents, sentences):
        init_len = len(init_sents)
        remained_len = len(sentences)
        filtered_ratio = 1 - remained_len / init_len
        if filtered_ratio > 0.66:
            print(
                f"[WARNING] > {init_len - remained_len}/{init_len} sentences were removed from the original generation.\n            We will omit its fact verifications.")
            sentences = []
        #
        f_filtered = False if init_len == remained_len else True

        return f_filtered, sentences

    def get_filtered_sentences(self, init_sents, topic, writer):
        sentences, verdicts = self.filter_abstained_sentences(topic, init_sents)
        writer.write({"topic": topic, "sentences": verdicts})
        #
        return self.determine_fully_filtered(init_sents, sentences)

    def run(self, topics, generations):
        def get_filtering_result(topic, results):
            verdicts = results[topic]
            init_sents = []
            sentences = []
            for s in verdicts.keys():
                init_sents.append(s)
                v = verdicts[s]
                if v:
                    sentences.append(s)
            #
            return self.determine_fully_filtered(init_sents, sentences)

        # read the cache file
        processed_topics = {}
        if os.path.exists(self.filter_cache_file):
            with jsonlines.open(self.filter_cache_file, mode='r') as reader:
                for l in reader:
                    processed_topics[l["topic"]] = l["sentences"]
        len_done_topics = len(processed_topics.keys())

        #
        with jsonlines.open(self.filter_cache_file, mode='a', flush=True) as writer:
            valid_topics = []
            valid_gens = []
            valid_length_pairs = []
            filtered_lengths = []
            for t, g in zip(topics, generations):

                if len_done_topics > 0 and t in processed_topics:
                    # get filtered sentences from a cache file
                    f_filtered, sentences = get_filtering_result(t, processed_topics)
                else:
                    # filtering using model
                    init_sents = self.get_init_sentences(g)
                    f_filtered, sentences = self.get_filtered_sentences(init_sents, t, writer)
                #
                len_init_gen = len(g)
                if len(sentences) == 0:
                    print(f"\t$$$$$\t [{t}] is invalid", flush=True)
                    filtered_lengths.append(len_init_gen)
                else:
                    new_gen = " ".join(sentences)
                    valid_topics.append(t)
                    valid_gens.append(new_gen)
                    if f_filtered:
                        len_new_gen = len(new_gen)
                        valid_length_pairs.append((len_init_gen, len_new_gen))
        #
        return valid_topics, valid_gens, valid_length_pairs, filtered_lengths

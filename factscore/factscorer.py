import argparse
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from factscore.atomic_facts import AtomicFactGenerator

from factscore.npm import NPM
from factscore.retrieval import DocDB, Retrieval
from factscore.auto_modelfactory import ModelFactory
import jsonlines
import pandas as pd
from factscore.inval_gen_filter import InvalidGenerationFilter

import sys
# from factscore.abstain_detection import is_response_abstained

class FactScorer(object):

    def __init__(self,
                 model_name_or_path = "gpt-3.5-turbo-0125",
                 data_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 api_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256,
                 af_model_name="gpt-3.5-turbo-0125",
                 retrieval_type="bm25",
                 retrv_k=5
                 ):
        self.sent_filter_demon_filename = "k_filter_demons_v1.json"
        self.sent_filter_lm_cache_filename = "1_sent_filter_cache.pkl"
        self.sent_filter_cache_filename = "1_sent_filter_results.jsonl"
        self.af_demon_filename = "k_demons_v1.json"
        self.af_lm_cache_filename = "2_af_cache.pkl" # originally, "InstructGPT.pkl"
        self.af_gen_cache_filename = "2_af_gen_results.jsonl"
        self.retrieval_cache_filename = f"3_retrieval-{retrieval_type}-k{retrv_k}"
        self.vry_model_cache_filename = f"4_{model_name_or_path.replace('/','-')}.pkl"
        self.decision_cache_filename = f"4_af_decisions_{retrieval_type}_k{retrv_k}.jsonl"

        self.af_model_name = af_model_name
        self.vry_model_name = model_name_or_path
        self.retrv_k = retrv_k
        self.retrieval_type = retrieval_type

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.api_key = api_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate
        
        self.lm = ModelFactory.from_pretrained(self.vry_model_name,
                        cache_file=os.path.join(cache_dir, self.vry_model_cache_filename),
                        key_path=api_key)

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="kowiki-20240301", db_path=None, data_path=None, retrieval_type='bm25'):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"{self.retrieval_cache_filename}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"{self.retrieval_cache_filename}.pkl")

        if 'cross-encoder' in self.retrieval_type:
            tokenizer_path = 'enter proper tokenizer_path'
            ckp_path = 'enter proper ckp_path' # cross_encoder 사용시
        else:
            tokenizer_path = None
            ckp_path = None

        self.db[name] = DocDB(db_path=db_path, data_path=data_path, tokenizer_path=tokenizer_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, retrieval_type=self.retrieval_type, batch_size=self.batch_size, ckp_path=ckp_path)


    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "kowiki-20240301"                # kmh

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.api_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, self.af_lm_cache_filename),
                                                        af_model_name=self.af_model_name, demon_fn=self.af_demon_filename)  # kmh

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            afc_fp = os.path.join(self.cache_dir, self.af_gen_cache_filename)
            if os.path.exists(afc_fp):
                with jsonlines.open(afc_fp) as reader:
                    for l in reader:
                        afs = l["atomic_facts"]
                        afs = [fact for _, facts in afs for fact in facts]
                        if len(afs) == 0:
                            atomic_facts.append(None)
                        else:
                            atomic_facts.append(afs)
                        if len(atomic_facts) % 10 == 0:
                            self.af_generator.save_cache()
            else:
                af_cache_writer = jsonlines.open(afc_fp, mode='w')
                #
                for i, (topic, gen) in enumerate(zip(topics, generations)):
                    curr_afs, _ = self.af_generator.run(gen)
                    #
                    af_cache_writer.write({"topic": topic, "generation": gen, "atomic_facts": curr_afs})  # kmh
                    #
                    curr_afs = [fact for _, facts in curr_afs for fact in facts]
                    if len(curr_afs) == 0:
                        atomic_facts.append(None)
                    else:
                        atomic_facts.append(curr_afs)
                    if len(atomic_facts) % 10 == 0:
                        self.af_generator.save_cache()
                #
                af_cache_writer.close()

            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        assert len(topics) == len(atomic_facts), f"topics({len(topics)}) and atomic_facts({len(atomic_facts)}) have different numbers"

        if verbose:
            topics = tqdm(topics)

        print('\n ----------------- verify atomic facts -----------------', flush=True)
        scores = []
        init_scores = []
        decisions = []
        d_path = os.path.join(self.cache_dir, self.decision_cache_filename) 
        if os.path.exists(d_path): # 파일이 있으면 읽어오기
            d_fp = jsonlines.open(d_path, mode='r')
            n_samples = 0
            for decision in d_fp:
                score = np.mean([d["is_supported"] for d in decision])
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(atomic_facts[n_samples])>gamma else np.exp(1-gamma/len(atomic_facts[n_samples]))
                    score = penalty * score
                decisions.append(decision)
                scores.append(score)
                n_samples += 1
        else:
            d_fp = jsonlines.open(d_path, mode='w')
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is None:
                    decisions.append(None)
                else:
                    decision = self._get_score(topic, generation, facts, knowledge_source)
                    score = np.mean([d["is_supported"] for d in decision])
                    
                    if gamma:
                        init_scores.append(score)
                        penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                        score = penalty * score
                    
                    decisions.append(decision)
                    scores.append(score)
                    if len(scores) % 10 == 0:
                        self.save_cache()
                    d_fp.write(decision)

            self.save_cache()
            d_fp.close()
        
        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        total_words = 0
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=self.retrv_k)
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nAnswer:".format(definition.strip(), atom.strip())

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue

                output = self.lm.generate(prompt)

                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    # Compatible only when using Llama 2 vocabulary
                    # assert logits.shape[0] in [32000, 32001]
                    if output[2] != 5852:
                        raise ValueError(f"Error: The 'True' token index {output[2]} does not match the expected True token index 5852.")
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                    if isinstance(is_supported, np.bool_):
                        is_supported = bool(is_supported)
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            decisions.append({"atom": atom, "is_supported": is_supported, "context": context})
            print(f"[{is_supported}] {atom}", flush=True)

        if cost_estimate:
            return total_words
        else:
            return decisions


def write_to_excel_file(out, in_path, out_dir, af_model_name, vry_model_name, retrv_k):
    _, in_fn = os.path.split(in_path)
    in_fn1, _ = os.path.splitext(in_fn)
    out_fn = f"5_afs_verification.xlsx"
    out_path = os.path.join(out_dir, out_fn)

    atoms = []
    verdicts = []
    contexts = []
    decisions = out["decisions"]
    for decision in decisions:
        for af_d in decision:
            atoms.append(af_d["atom"])
            verdicts.append(af_d["is_supported"])
            contexts.append(af_d["context"])
        atoms.append("-")
        verdicts.append("-")
        contexts.append("-")

    # write to an EXCEL file
    df = pd.DataFrame({"atom": atoms, "sys.verdict": verdicts, "context": contexts})
    df.to_excel(out_path, index=False)


def do_main():
    parser = argparse.ArgumentParser()
    # gpt-4-turbo-preview(gpt-4-0125-preview)     gpt-3.5-turbo-0125            gpt-3.5-turbo-instruct
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default="gpt-3.5-turbo-0125")
    parser.add_argument('--af_model_name',
                        type=str,
                        default="gpt-3.5-turbo-0125")
    parser.add_argument('--filter_temperature',
                        type=float,
                        default=1.0,        # 2024/07/10 기준 temperature는 [0,2] 구간의 값이 허용됨. default = 1.0
                        help="temperature for filter of invalid generation")
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--api_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default="kowiki-20240301")


    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('-max_n_samples',
                        type=int,
                        default=None)

    parser.add_argument('--retrv_k',
                        type=int,
                        default=5) # # of retrieval samples
    parser.add_argument('--retrieval_type',
                        type=str,
                        default='bm25',
                        choices=["bm25", "gtr-t5-large","cross-encoder"])
    # TODO: overwrite cache directory 기능 추가

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(data_dir=args.data_dir,
                    model_name_or_path=args.model_name_or_path,
                    cache_dir=args.cache_dir,
                    api_key=args.api_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type,
                    af_model_name=args.af_model_name,
                    retrieval_type=args.retrieval_type, 
                    retrv_k=args.retrv_k) 

    tot = 0
    cnt_valid_topics = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:       # TODO(kmh) remove??
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."

                annotations = dp["annotations"]
                if len(annotations) > 0:
                    atomic_facts.append([atom["text"] for sent in annotations for atom in sent["model-atomic-facts"]])
                    topics.append(dp["topic"])
                    generations.append(dp["output"])
                    cnt_valid_topics += 1
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.max_n_samples is not None and tot==args.max_n_samples:
                break
    # topic(인물)에 대한 실질적인 내용을 담고 있지 않는 문장(Invalid Generation)을 제거함
    demon_dir = os.path.join(args.data_dir, "demos")
    if args.use_atomic_facts == False:      # kmh@240825
        igfilter = InvalidGenerationFilter(key_path=args.api_key,            # TODO(kmh) unify the key file
                                           filter_demon_file=os.path.join(demon_dir, fs.sent_filter_demon_filename),
                                           lm_cache_file=os.path.join(args.cache_dir, fs.sent_filter_lm_cache_filename),
                                           filter_cache_file=os.path.join(args.cache_dir, fs.sent_filter_cache_filename),
                                           model_name=args.af_model_name, #"gpt-3.5-turbo-0125"
                                           temperature=args.filter_temperature)
        topics, generations, valid_length_pairs, filtered_lengths = igfilter.run(topics, generations)
        cnt_valid_topics = len(topics)

    #
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    #
    out["respond_ratio"] = cnt_valid_topics/tot
    logging.critical("Successful respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    #
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # convert Binary variable into string for json.dumps()
    for d in out['decisions']:
        for a in d:
            if a['is_supported']:
                a['is_supported']="true"
            else:
                a['is_supported']="false"

    write_to_excel_file(out, args.input_path, fs.cache_dir, fs.af_model_name, fs.vry_model_name, args.retrv_k)

    # Save out as a json file
    with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
        f.write(json.dumps(out,ensure_ascii = False) + "\n")


if __name__ == '__main__':
    do_main()
    print(f"\n\n\t ########## END: __main__ ##########")



import argparse
import string
import json
import numpy as np
import os
import logging
from tqdm import tqdm
import jsonlines
import pandas as pd
import sys

from factscore.atomic_facts import AtomicFactGenerator
from factscore.inval_gen_filter import InvalidGenerationFilter
from factscore.retrieval import DocDB, Retrieval
from factscore.auto_modelfactory import ModelFactory
# server related
import socket
import threading
import openai


class FactScorer(object):

    def __init__(self,
                 args,
                 model_name_or_path="gpt-3.5-turbo-0125",
                 data_dir="downloaded_files",
                 cache_dir=".cache/factscore",
                 api_key="api.keys",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256,
                 af_model_name="gpt-3.5-turbo-0125",
                 filter_temperature="1.0",
                 retrieval_type="bm25",
                 retrv_k=5
                 ):
        self.sent_filter_demon_filename = "k_filter_demons_v1.json"
        self.sent_filter_lm_cache_filename = "1_sent_filter_cache.pkl"
        self.sent_filter_cache_filename = "1_sent_filter_results.jsonl"
        self.af_demon_filename = "k_demons_v1.json"
        self.af_lm_cache_filename = "2_af_cache.pkl" # originally, "InstructGPT.pkl"
        self.af_gen_cache_filename = "2_af_gen_results.jsonl"
        self.af_filter1_cache_filename = "2-2_af_filter_results.jsonl"
        self.af_filter1_lm_cache_filename = "2-2_af_filter_cache.jsonl"
        self.retrieval_cache_filename = f"3_retrieval-{retrieval_type}-k{retrv_k}"
        self.vry_model_cache_filename = f"4_{model_name_or_path.replace('/','-')}.pkl"
        self.decision_cache_filename = f"4_af_decisions_{retrieval_type}_k{retrv_k}.jsonl"

        self.filter_temperature = filter_temperature

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
        self.filter_af = args.filter_atomic_facts
        self.cost_estimate = cost_estimate

        print(f"@FactScorer", flush=True)
        self.lm = ModelFactory.from_pretrained(self.vry_model_name,
                        cache_file=os.path.join(cache_dir, self.vry_model_cache_filename),
                        key_path=api_key)
        # kmh
        self.knowledge_source = args.knowledge_source
        self.gamma = args.gamma

    def setup_db_and_af_generator(self, knowledge_source=None):
        if knowledge_source is None:    # then, use the default knowledge source
            knowledge_source = "kowiki-20240301"
        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)
        #
        self.af_generator = AtomicFactGenerator(key_path=self.api_key,
                                                demon_dir=os.path.join(self.data_dir, "demos"),
                                                gpt3_cache_file=os.path.join(self.cache_dir, self.af_lm_cache_filename),
                                                af_model_name=self.af_model_name,
                                                demon_fn=self.af_demon_filename)

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
            tokenizer_path = 'reranking_lce_202402/MoBERT-Large/'
            ckp_path = 'reranking_lce_202402/trained_model/law_wiki_etri_2e-5' # cross_encoder 사용시
        else:
            tokenizer_path = None
            ckp_path = None

        self.db[name] = DocDB(db_path=db_path, data_path=data_path, tokenizer_path=tokenizer_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, retrieval_type=self.retrieval_type, batch_size=self.batch_size, ckp_path=ckp_path)

    def get_score1(self, _client, topic, generation, knowledge_source=None,
                   gamma=10,
                   atomic_facts=None,
                   verbose=False):

        # AF 생성
        atomic_facts = []
        curr_afs, _ = self.af_generator.run(generation)
        curr_afs = [fact for _, facts in curr_afs for fact in facts]
        #
        if len(curr_afs) == 0:
            return True     # means 'Filtered'

        print('\n ----------------- verify atomic facts -----------------', flush=True)
        # else:
        # for topic, generation, facts in zip(topics, generations, atomic_facts):
        decision = self._get_score1(_client, topic, generation, curr_afs, knowledge_source)
        score = np.mean([d["is_supported"] for d in decision])
        print(f"\tinitial factscore: {score}", flush=True)
        if decision is not None: print(f"\tnum_facts: {len(decision)}", flush=True)

        #
        return False        # means 'Good'


    def _get_score1(self, _client, topic, generation, atomic_facts, knowledge_source):
        def make_prompt(topic, atom, retrv_k, knowledge_source):
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=retrv_k)
            definition = "Answer the question about {} based on the given context.\n\n".format(topic)
            context = ""
            for psg_idx, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"],
                                                            psg["text"].replace("<s>", "").replace("</s>", ""))
            definition += context.strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            prompt = "{}\n\nInput: {} True or False?\nAnswer:".format(definition.strip(), atom.strip())

            return prompt, context

        def decide_verdict(output):
            if type(output[1]) == np.ndarray:
                # when logits are available
                logits = np.array(output[1])
                # $$$$$$$$$$ Compatible only when using "Llama 2" vocabulary $$$$$$$$$$
                # $$$$$$$$$$ Compatible only when using "Llama 2" vocabulary $$$$$$$$$$
                # $$$$$$$$$$ Compatible only when using "Llama 2" vocabulary $$$$$$$$$$
                assert logits.shape[0] in [32000, 32001]
                if output[2] != 5852:
                    raise ValueError(
                        f"Error: The 'True' token index {output[2]} does not match the expected True token index 5852.")
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
                    is_supported = all([keyword not in generated_answer.lower().translate(
                        str.maketrans("", "", string.punctuation)).split() for keyword in
                                        ["not", "cannot", "unknown", "information"]])

            return is_supported

        # ########## core body ########## #
        decisions = []
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                #
                prompt, context = make_prompt(topic, atom, self.retrv_k, knowledge_source)
                output = self.lm.generate(prompt)
                is_supported = decide_verdict(output)   # decision(lm의 결과(output) 해석)
            else:
                print(f"\n\t[ERROR] Verifier(self.lm) is None!!!")
                is_supported = True
                context = ''

            #
            result = {"atom": atom, "is_supported": is_supported} # , "context": context
            decisions.append(result)
            print(f"[{is_supported}] {atom}", flush=True)
            # for ui
            result_json = json.dumps(result) + "\n"

        return decisions

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

        #
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


def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--host_ip', type=str, default="129.254.169.68")
    parser.add_argument('--port_no', type=str, default="9995")
    #
    # gpt-4-turbo-preview(gpt-4-0125-preview)     gpt-3.5-turbo-0125            gpt-3.5-turbo-instruct
    parser.add_argument('--model_name_or_path', type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument('--af_model_name', type=str, default="gpt-3.5-turbo-0125")
    # 2024/07/10 기준 temperature는 [0,2] 구간의 값이 허용됨. default = 1.0
    parser.add_argument('--filter_temperature', type=float, default=1.0,
                        help="temperature for filter of invalid generation")
    parser.add_argument('--filter_atomic_facts', action="store_true")
    parser.add_argument('--input_path', type=str,
                        default="ui_input.json")
    parser.add_argument('--gamma', type=int, default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--api_key', type=str, default="api.keys")
    parser.add_argument('--data_dir', type=str, default="downloaded_files")
    parser.add_argument('--cache_dir', type=str, default=".cache/factscore/")
    parser.add_argument('--knowledge_source', type=str, default="kowiki-20240301")

    parser.add_argument('--cost_estimate', type=str, default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type', type=str, default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts', action="store_true")
    parser.add_argument('--verbose', action="store_true", help="for printing out the progress bar")
    parser.add_argument('--print_rate_limit_error', action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--max_n_samples', type=int, default=None)

    parser.add_argument('--retrv_k', type=int, default=5) # # of retrieval samples
    parser.add_argument('--retrieval_type', type=str, default='bm25',
                        choices=["bm25", "gtr-t5-large","cross-encoder"])
    # TODO: overwrite cache directory 기능 추가

    args = parser.parse_args()

    return args


###########################################################

logger = logging.getLogger(__name__)


def print_warning_message(msg):
    print(f"{msg}", flush=True)
    logger.info(msg)


def check_topic_is_valid(db, topic):
    passages = db.get_text_from_title(topic)
    if len(passages) == 0:
        return False
    else:
        return True

def check_openai_api_key(api_key):
    # OpenAI API 키 설정
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)

    try:
        # 간단한 API 호출 (예: 모델 목록 가져오기)로 키 유효성 확인
        models = client.models.list()
        print("API key is valid.")
        with open("api.keys", "w") as writer:
            writer.write(f"openai={api_key}")
        return True, ""
    # except openai.InvalidRequestError:
    #     return False, "Invalid API key."
    except Exception as e:
        return False, f"An error occurred: {e}"


def initialize_factscorer(args):
    fs = FactScorer(args,
                    data_dir=args.data_dir,
                    model_name_or_path=args.model_name_or_path,
                    cache_dir=args.cache_dir,
                    api_key=args.api_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type,
                    af_model_name=args.af_model_name,
                    filter_temperature=args.filter_temperature,
                    retrieval_type=args.retrieval_type,
                    retrv_k=args.retrv_k)
    fs.setup_db_and_af_generator(args.knowledge_source)

    demon_dir = os.path.join(args.data_dir, "demos")
    igfilter = InvalidGenerationFilter(key_path=args.api_key,
                                       filter_demon_file=os.path.join(demon_dir, fs.sent_filter_demon_filename),
                                       lm_cache_file=os.path.join(args.cache_dir, fs.sent_filter_lm_cache_filename),
                                       filter_cache_file=os.path.join(args.cache_dir, fs.sent_filter_cache_filename),
                                       model_name=args.af_model_name,  # "gpt-3.5-turbo-0125"
                                       temperature=args.filter_temperature)

    return fs, igfilter


def calculate_factscore(fs, igfilter, input_json, _client):
    #
    topic = input_json["topic"]
    generation = input_json["output"]
    # 검증 비대상 문장 필터링
    f_filtered, new_gen = igfilter.run1(topic, generation)
    new_gen = new_gen.strip()
    if not f_filtered or len(new_gen) > 0:
        # 실제로 돌아가는 부분
        f_filtered = fs.get_score1(_client, topic, new_gen, knowledge_source=fs.knowledge_source, gamma=fs.gamma, atomic_facts=None)

    if f_filtered:
        print_warning_message("[WARNING] 사실성 평가에 부적합한 문서입니다.\n사실을 담고 있는 생성문을 입력해주세요.")

def do_main2():
    # #
    # with open(input_path, 'r', encoding='utf-8') as file:
    #     input_data = json.load(file)

    args = parse_args()
    #
    with open(args.input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)


    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)
    #
    model_init_result = initialize_factscorer(args)
    #
    f_valid, err_msg = check_openai_api_key(input_data["api_key"])
    #
    if f_valid:
        fs, igfilter = model_init_result
        topic = input_data["topic"]
        # with self.critical_section:
        f_valid_final = check_topic_is_valid(fs.retrieval[fs.knowledge_source].db, topic)
        _client = "dummy"
        if f_valid_final:
            calculate_factscore(fs, igfilter, input_data, _client)
        else:
            err_msg = f"[ERROR] {topic}을 위키백과에서 확인할 수 없습니다.\n다시 확인해주세요."
    else:
        print(err_msg, flush=True)
        err_msg = "[ERROR] 제공한 API key가 올바르지 않습니다.\n다시 확인해주세요."
        f_valid_final = False

    #
    if not f_valid_final:
        print_warning_message(err_msg)
    #
    with open("api.keys", "w") as writer:
        writer.write(f"openai=qwert\ngemini=yuiop\n")


if __name__ == '__main__':
    # input_path = 'ui_input.json'
    # do_main2(input_path)
    do_main2()

    print(f"\n\n\t ########## END: __main__ ##########")



import json
import numpy as np
import re
import string
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import os
import kss

# from factscore.openai_lm import OpenAIModel
from factscore.auto_modelfactory import ModelFactory

import functools
import spacy
import sys
import nltk
import openai
import time
from nltk.tokenize import sent_tokenize
import random
# nltk.download("punkt")          # kmh: blocked


def sent_tokenize_kor(paragraph):
    result = kss.split_sentences(paragraph)
    if len(result) == 0:
        return []
    return result


class AtomicFactGenerator(object):
    def __init__(self, key_path, demon_dir, gpt3_cache_file=None, af_model_name="gpt-3.5-turbo-0125", demon_fn="k_demons_v1.json"):
        # self.at_model_type = "ChatGPT"          # TODO(kmh) : remove
        self.prompt_prefix = "다음 문장을 독립적인 개별 사실을 표현하는 문장들로 분할해주세요: "  # kmh
        self.is_bio = True
        self.af_model_name = af_model_name
        self.demon_path = os.path.join(demon_dir, demon_fn)

        print(f"@AtomicFactGenerator", flush=True)
        self.openai_lm = ModelFactory.from_pretrained(af_model_name, cache_file=gpt3_cache_file, key_path=key_path)

        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_cache(self):
        self.openai_lm.save_cache()

    def run(self, generation, cost_estimate=None):
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None."""
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]

        # filtering for "gemini"'s results
        if "gemini" in self.af_model_name:
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

        return self.get_atomic_facts_from_paragraph(paragraphs, cost_estimate=cost_estimate)


    def get_atomic_facts_from_paragraph(self, paragraphs, cost_estimate=None):
        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0 :
                para_breaks.append(len(sentences))

            curr_sentences = sent_tokenize_kor(paragraph)
            # [BigFix] kmh@240807 BUG가 확인됨. 문제를 일으킴 그래서 block 시킴. 나중에 다시 개발해 보면 좋을 듯함  # TODO
            # initials = detect_initials(paragraph)
            # if len(initials) > 0:
            #     curr_sentences = fix_sentence_splitter(curr_sentences, initials)        # TODO(kmh):  실제로 동작을 하는지 확인해볼 필요가 있음.

            sentences += curr_sentences

        # TODO(kmh) remove: 영어권에서만 적용 가능
        # filtered_sentences = [sent for i, sent in enumerate(sentences) if not (not self.is_bio and ( \
        #                     (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
        #                     (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))))]
        # atoms_or_estimate = self.get_init_atomic_facts_from_sentence(filtered_sentences, cost_estimate=cost_estimate)
        atoms_or_estimate = self.get_init_atomic_facts_from_sentence(sentences, cost_estimate=cost_estimate)

        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate


        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            f_found_invalid = False
            # TODO(kmh): af sentence filtering : 불필요한 AF 문장 제거
            ## HERE: code for filtering
            # if not self.is_bio and ( \
            #     (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
            #     (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))):
            #     atomic_facts_pairs.append((sent, []))
            # elif self.is_bio and sent.startswith("This sentence does not contain any facts"):
            #     atomic_facts_pairs.append((sent, []))
            # elif sent.startswith("Sure") or sent.startswith("Please") or (i==0 and sent.startswith("Here are")):
            #     atomic_facts_pairs.append((sent, []))
            # else:
            atomic_facts_pairs.append((sent, atoms[sent]))

        return atomic_facts_pairs, para_breaks


    def get_init_atomic_facts_from_sentence(self, sentences, cost_estimate=None):
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None."""

        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for sentence in tqdm(sentences, desc="Generating AFs >> "):
            if sentence in atoms:
                continue

            prompt = ""
            top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            for match in top_machings:
                prompt = prompt + f"{self.prompt_prefix} {match}\n"
                for fact in demons[match]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            prompt = prompt + f"{self.prompt_prefix} {sentence}\n"
            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            for prompt in prompts:
                output, _ = self.openai_lm.generate(prompt)
                atoms[prompt_to_sent[prompt]] = text_to_sentences(output)         # org
                # atoms[prompt_to_sent[prompt]] = text_to_sentences_v240423(output)           # kmh

            for key, value in demons.items():
                if key not in atoms:
                    atoms[key] = value

            return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# transform InstructGPT output into sentences
def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences


def text_to_sentences_v240423(text):        # for the current openai's API
    sentences = text.split("\n")
    new_sentences = []
    for i, sent in enumerate(sentences):
        sub_sents = sent.split(".")
        new_sent = None
        if len(sub_sents[1]) == 0:
            print(f"\t $$$$$ [WARNING] in sent : \n\t\t sent : {sent} \n\t\t new_sent : {new_sent} ")
        if isinstance(sub_sents, list):
            first_item = sub_sents[0]
            if isinstance(first_item, str):
                if first_item.isdigit():
                    new_sent = sub_sents[1]
        else:
            new_sent = sent.strip()
        if new_sent != None:
            new_sent = new_sent.strip()
            if new_sent[-1] != '.':
                new_sent = new_sent + '.'
            new_sentences.append(new_sent)

    return new_sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):          # kmh: no longer used

    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

# TODO(kmh) : 새로운 한글 example로 교체
def main():
    generator = AtomicFactGenerator("api.key", "demos", gpt3_cache_dir=None)
    atomic_facts, para_breaks = generator.run("Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.\n\nHenry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.")

    print(atomic_facts)
    print(para_breaks)

if __name__ == "__main__":
    main()
# �ѱ��� ���� ������ ���� ��� ��� ��� ���� ���� ���

KorFactScore (Korean Factual precision in atomicity Score)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

### ���� ���

* �ΰ�����(AI)�� ������ ����?�� ������ �ŷ��� �� �ִ� ���� ���� �߰����� ������ �ʿ���.
* ������ �ѱ��� ������ ���� ���� ������ ��� ���踦 �����ϰ� ������ �� �ִ� ������ ������.
* Ư��, �ֱ� �پ��� �ѱ��� AI ��� �𵨵�� �뷮�� �������� �����ϰ� �ִ� ��Ȳ���� �̷��� ����� �ʿ伺�� ���� �����ǰ� ����.

### �ֿ� Ư¡

* �ѱ��� ���� ������ ����� ���� ��ǵ�(Atomic?facts)�� ���� ��� ���踦 �����ϴ� ����� ����
* Ư�� ��� �𵨿� ���ѵ��� ������(Model Agnostic), �򰡿� ������ ��ü�Ͽ� ���� ������ ���� �� ����
* �򰡸� ���� �����ͼ� ����: ������ ����, ���� ���(atomic facts)�� ���ҵ� ����� ����
* ���� �򰡸� ���� ���� ������(ground truth label)�� �� �ڵ带 �Բ� ����

#### ���� ���(Atomic facts)
* �ǹ�: ������ �ٽ� �ǹ̸� ��� �ִ� ���� ���� ����
* ����: ���� ������ 3���� ���� ��Ƿ� ���ص� �� ����
    * "ȫ������ ���� �ı��� �������� ��ġ���̴�."
�� ["ȫ������ ���� �ı��� ����̴�.", "ȫ������ �����̴�.", "ȫ������ ��ġ���̴�."]
<br>
<br>

## ���� ȯ�� ��ġ

`conda` Ȥ�� `virtualenv`�� Ȱ���� Python 3.9+ ȯ���� �����ϼ���. (google gemini API�� ����Ϸ��� Python 3.9 �̻��� �ʿ��մϴ�.)

* �ʼ� ��ġ ��Ű���� [requirements.txt](https://github.com/ETRI-XAINLP/KorFactScore/blob/master/requirements.txt)�� Ȯ���غ�����.

```bash
conda create -n korfs-env python=3.9
conda activate korfs-env

git clone https://github.com/ETRI-XAINLP/KorFactScore.git
cd KorFactscore
pip install -r requirements.txt
```

### Huggingface LLaMa model �ٿ�ε� ���

1. huggingface�� ������ ����, User Access Tokens�� ��û�ϰ� �޾ƵӴϴ�.
2. �͹̳ο��� huggingface �α����� �����մϴ�. (�ɼ�) `model_cache_dir`���� ���� �ٿ�ε� �Ǿ� �޾��� ������ ��ġ�� �̸� �����صθ� �˴ϴ�.

```bash
huggingface-cli login
export HF_HOME={model_cache_dir} # optional
```

3. `transformers`�� ��ġ�� env �Ʒ����� huggingface�� LLaMa ���� �ٿ�ε� �մϴ�. �ڼ��� ������ [LLaMa Ȩ������](https://github.com/meta-llama/llama3)�� �����ϼ���.

### �ϵ���� ���ѻ���

* �򰡿� ������ ������ �� �־�� �մϴ�. (�򰡿� ���� ����: ChatGPT API, Gemini API, LLaMa-2, LLaMa-3, LLaMa-3.1 ��)

<br>
<br>

## KorFactScore ���� ���

```bash
python -m factscore.factscorer \
    --input_path {input_path} \
    --af_model_name {af_generator_name} \
    --model_name_or_path {verifier_name} \
    --retrieval_type {retrieval_name} \
    --api_key {api_keys}
```

* `--input_path` �� `data/k_unlabeled/gen_bio-gpt_4-kr-all.jsonl`�� ���� �����Դϴ�. `jsonlines` format���� �� ���� ���� `topic`�� `output` (���� ������ ���)�� �����մϴ�.
* `--data_dir`: �˻��⿡ ���� knowledge source�� ���丮 (Default ����: `.cache/factscore`)
* `--cache_dir`: ���α׷� ���� ������� ����Ǵ� ���丮 (Default ����: `.cache/factscore`)
* `--af_model_name`: atomic facts�� ������ �� ���� �� �̸��� �ֽ��ϴ�. `gpt-3.5-turbo-0125`, `gpt-4-0125-preview` Ȥ�� `gemini-1.0-pro` ���� �����մϴ�.
* `--model_name_or_path`: atomic facts�� �����ϴ� ���� ���� �̸� Ȥ�� ��ġ�� �ֽ��ϴ�. af_model_name�� ���������� `gpt-3.5-turbo-0125`, `gpt-4-0125-preview` Ȥ�� `gemini-1.0-pro` Ȥ�� `meta-llama/llama-2-` ���� �����մϴ�.
* `--api_key`: OpenAI API Key �Ǵ� Google Gemini API Key�� �����ϴ� ������ �̸��Դϴ�. ����� API�� ���� key ���� �־�μ���. (�Ʒ� ����)

    ```txt
    openai=abcdefg
    gemini=xyjklmn
    ```

\*\* ����ϴ� �𵨿� ���� API ��� ����� û���� �� �ֽ��ϴ�.

**Optional flags**:

* `--use_atomic_facts`: �� flag �����, atomic fact generator�� ������� �ʰ� �̸� ������ atomic facts�� �����ͼ� ����� �� �ֽ��ϴ�. �� ��� `input_path`�� atomic facts�� �����ϰ� �ִ� �Է� �����̾�� �մϴ�. ������ `data/k_labeled/gen_bio-gpt_4-{kr/fr}-all.jsonl` ������ gpt-4�� ������ atomic-facts�� �����ϰ� �ֽ��ϴ�. (�Ʒ� ���� ����� ������ �� ���˴ϴ�.)
* `--knowledge_source`: �⺻������ kowiki-20240301.db�� �����Ǿ� �ֽ��ϴ�. (�ѱ��� Wikipedia - 2024/03/01).

<!--���� DB�� �����ϰ� �ʹٸ� [�Ʒ� ����](#To-use-a-custom-knowledge-source) ����� �����ϼ���. �׸��� db �̸��� �� flag�� �־��ּ���. -->

<br>
<br>

## ����� ������

* �ι��� ����� GPT-4�� ������ ������� �̷���� �����ͼ�
    * ��ġ:
        * data/k_unlabeled/
            (atomic facts ������)
    * �����̸�:
        * gen_bio-gpt_4-kr-all.jsonl
        * gen_bio-gpt_4-fr-all.jsonl

    | ������ | ���� �� |
    | --- | ---- |
    | �ѱ��� ��� | 64 |
    | �ܱ��� ��� | 50 |

<br>
<br>

## ���� �� ���

### ���� �򰡿� ������

* ����� �����Ϳ� atomic facts ���� ����� �߰��� �����ͼ�.
    Atomic facts ������ ���� ���� gpt-4-turbo �Դϴ�.
    * ��ġ:
        * data/k_labeled/
            (atomic facts ����)
    * �����̸�:
        * gen_bio-gpt_4-kr-all.jsonl
        * gen_bio-gpt_4-fr-all.jsonl

### �����⿡ ���� ��

* OpenAI Models
    * gpt-3.5-turbo-0125
    * gpt-4-0125-preview
* Google Models
    * gemini-1.0-pro
    * gemini-1.5-pro
* LLaMa Models
    * llama-2 models
    * llama-3 models
    * llama-3.1 models
* Others
    * LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct

### System�� ����Ǵ� ���� ��� ���

����(ground truth)�� system�� ��� ������ ��ġ�ϴ� ������ ����Ͽ� ����
* ���� ������ġ �� �̸�:
    * data/k_truth_annotations/gen_bio-gpt_4-kr-all-gold.jsonl
    * data/k_truth_annotations/gen_bio-gpt_4-fr-all-gold.jsonl

```bash
human_judge_data='data/k_truth_annotations/gen_bio-gpt_4-kr-all-gold.jsonl'
system_decisions_data='cache/gpt_4-kr-all/4_af_decisions_bm25_k5.jsonl'

python factscore/evaluate_system_vs_human_judgments ${human_judge_data} ${system_decisions_data}
```

* �˻���: BM25
</br>

----
\* �Ʒ� ���� ����� [KorFactScore ���� ���](#korfactscore-����-���)�� ���� ���� �����մϴ�.
### ��� (1) System�� ����Ǵ� ����

* `System�� ����Ǵ� ����`�� System ����� ����(ground truth)�� ��ġ�� ������ �����ݴϴ�(����: %).

| Model | Size | �ѱ��� <br> BM25 |  Cross-Encoder | �ܱ��� <br> BM25 |  Cross-Encoder |
| ----- | ---- | --------- | -------------- | --------- | -------------- |
| GPT-3.5 | - | 82.4 | 85.4 | 75.1 | 77.2 |
| GPT-4 | - | **92.4** | <span style="color:#e11d21;">**94.2**</span> | **91.6** | <span style="color:#e11d21;">**92.8**</span> |
| Gemini-1.0 | - | 81.4 | 84.5 | 68.2 | 79.3 |
| Gemini-1.5 | - | 81.9 | 57.6 | 70.3 | 74.2 |
| EXAONE-7.8B | 7.8B | 89.9 | 90.8 | 86.6 | 86.0 |
| LLaMa-2-chat | 7B | 61.1 | 64.5 | 60.1 | 62.8 |
|  | 13B | 70.6 | 71.5 | 80.6 | 80.7 |
|  | 70B | 82.3 | 82.9 | 76.3 | 79.7 |
| LLaMa-3-Instruct | 8B | 84.7 | 86.2 | 83.9 | 86.5 |
|  | 70B | 89.1 | 90.7 | 86.5 | 86.9 |
| LLaMa-3.1-Instruct | 8B | 78.6 | 82.0 | 69.6 | 70.9 |
|  | 70B | 89.7 | 90.3 | 89.9 | 90.2 |

\* `Cross-Encoder`�� ETRI ���� ����� ���ߵ� �˻���μ� ��������� ���� ����� �����մϴ�.

### ��� (2) ���������� ��Ǽ� ���� (FactScore)

* `���������� ��Ǽ� ����`�� �Էµ� ������ ���� ��ǵ�(atomic facts)�� ���� system�� ���� ��Ǽ� ����, �� FactScore�Դϴ�(����: %).
    (Ground truth�� ��Ǽ� ������ �ѱ��� ��� 46.8, �ܱ��� ��� 80.1 �Դϴ�.)

| Model | Size | �ѱ��� <br> BM25 |  Cross-Encoder | �ܱ��� <br> BM25 |  Cross-Encoder |
| ----- | ---- | --------- | -------------- | --------- | -------------- |
| GPT-3.5 | - | 45.3 | 47.4 | 62.9 | 65.4 |
| GPT-4 | - | 53.8 | 53.1 | 79.6 | 81.1 |
| Gemini-1.0 | - | 33.4 | 36.2 | 55.2 | 64.1 |
| Gemini-1.5 | - | 52.0 | 70.0 | 81.0 | 85.3 |
| EXAONE-7.8B | 7.8B | 50.7 | 51.2 | 76.3 | 76.1 |
| LLaMa-2-chat | 7B | 39.6 | 37.3 | 57.6 | 59.6 |
|  | 13B | 64.8 | 67.3 | 79.0 | 80.3 |
|  | 70B | 48.9 | 49.8 | 67.1 | 69.9 |
| LLaMa-3-Instruct | 8B | 48.9 | 50.9 | 73.8 | 77.4 |
|  | 70B | 47.9 | 47.3 | 74.5 | 74.3 |
| LLaMa-3.1-Instruct | 8B | 27.9 | 31.1 | 52.8 | 54.6 |
|  | 70B | 54.7 | 54.5 | 82.0 | 82.7 |

<!-- ����? ### ���ѻ��� 
* ������ ���� ���ĺ��̽��� ����� ��Ű����� ������. ������ ������ ���ĺ��̽��� ������ ��� Ȯ�� ����. -->

## Citation

* �ѱ��� ���� ������ ���� ��� ���迡 ���� ���� ��� [Online] [https://github.com/ETRI-XAINLP/KorFactScore](https://github.com/ETRI-XAINLP/KorFactScore)
* **[Funding]** (4����) �������� ��� �Ǵܰ���� ����/�ٰŸ� �������� ������ �ǻ���� ���� �ΰ����� ��� ����
    * ����߽��ΰ������ٽ� ��õ������� ��� > ����� ������ �÷��׾��÷��� ����� �����ɼ� ���� ��� ����
        \(���б��������ź� ��� \| ������ű�ȹ�򰡿� ����\)

</br>
# 한국어 생성 문서의 원소 사실 기반 사실 관계 설명 기술

KorFactScore (Korean Factual precision in atomicity Score)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

### 개발 배경

* 인공지능(AI)이 생성한 문서 내 정보가 신뢰할 수 있는 지에 대한 추가적인 검증이 필요함.
* 생성된 한국어 문서에 대해 원소 단위로 사실 관계를 검증하고 설명할 수 있는 도구가 부재함.
* 특히, 최근 다양한 한국어 AI 언어 모델들로 대량의 문서들을 생성하고 있는 상황에서 이러한 기술의 필요성이 더욱 강조되고 있음.

### 주요 특징

* 한국어 생성 문서에 기술된 원소 사실들(Atomic facts)에 대한 사실 관계를 설명하는 기술을 제공
* 특정 언어 모델에 제한되지 않으며(Model Agnostic), 평가용 언어모델을 교체하여 모델의 성능을 평가할 수 있음
* 평가를 위한 데이터셋 제공: 생성된 문서, 원소 사실(atomic facts)로 분할된 문장들 포함
* 성능 평가를 위한 정답 데이터(ground truth label)와 평가 코드를 함께 제공

    #### 원소 사실(Atomic facts)
    * 의미: 문장의 핵심 의미를 담고 있는 작은 정보 단위
    * 예시: 다음 문장은 3개의 원소 사실로 분해될 수 있음
        * "홍인한은 조선 후기의 문인이자 정치가이다."
→ ["홍인한은 조선 후기의 사람이다.", "홍인한은 문인이다.", "홍인한은 정치가이다."]
<br>
<br>

## 구동 환경 설치

`conda` 혹은 `virtualenv`를 활용해 Python 3.9+ 환경을 설정하세요. (google gemini API를 사용하려면 Python 3.9 이상이 필요합니다.)

* 필수 설치 패키지는 [requirements.txt](https://github.com/ETRI-XAINLP/KorFactScore/blob/master/requirements.txt)를 확인해보세요.

```bash
conda create -n korfs-env python=3.9
conda activate korfs-env

git clone https://github.com/ETRI-XAINLP/KorFactScore.git
cd KorFactscore
pip install -r requirements.txt
```

### Huggingface LLaMa model 다운로드 방법

1. huggingface에 가입한 다음, User Access Tokens을 신청하고 받아둡니다.
2. 터미널에서 huggingface 로그인을 실행합니다. (옵션) `model_cache_dir`에는 모델이 다운로드 되어 받아질 폴더의 위치를 미리 지정해두면 됩니다.

```bash
huggingface-cli login
export HF_HOME={model_cache_dir} # optional
```

3. `transformers`가 설치된 env 아래에서 huggingface의 LLaMa 모델을 다운로드 합니다. 자세한 사항은 [LLaMa 홈페이지](https://github.com/meta-llama/llama3)를 참고하세요.

### 하드웨어 제한사항

* 평가용 언어모델의 구동 제한 사항을 따릅니다. (평가용 언어모델 예시: ChatGPT API, Gemini API, LLaMa-2, LLaMa-3, LLaMa-3.1 등)

<br>
<br>

## KorFactScore 실행 방법

```bash
python -m factscore.factscorer \
    --input_path {input_path} \
    --af_model_name {af_generator_name} \
    --model_name_or_path {verifier_name} \
    --retrieval_type {retrieval_name} \
    --api_key {api_keys}
```

* `--input_path` 는 `data/k_unlabeled/gen_bio-gpt_4-kr-all.jsonl`와 같은 형식입니다. `jsonlines` format으로 각 라인 별로 `topic`과 `output` (모델이 생성한 결과)을 포함합니다.
* `--data_dir`: 검색기에 사용될 knowledge source의 디렉토리 (Default 설정: `.cache/factscore`)
* `--cache_dir`: 프로그램 실행 결과값이 저장되는 디렉토리 (Default 설정: `.cache/factscore`)
* `--af_model_name`: atomic facts를 생성할 때 사용될 모델 이름을 넣습니다. `gpt-3.5-turbo-0125`, `gpt-4-0125-preview` 혹은 `gemini-1.0-pro` 등이 가능합니다.
* `--model_name_or_path`: atomic facts를 검증하는 검증 모델의 이름 혹은 위치를 넣습니다. af_model_name과 마찬가지로 `gpt-3.5-turbo-0125`, `gpt-4-0125-preview` 혹은 `gemini-1.0-pro` 혹은 `meta-llama/llama-2-` 등이 가능합니다.
* `--api_key`: OpenAI API Key 또는 Google Gemini API Key를 포함하는 파일의 이름입니다. 사용할 API의 개인 key 값을 넣어두세요. (아래 예시)

    ```txt
    openai=abcdefg
    gemini=xyjklmn
    ```

\*\* 사용하는 모델에 따라 API 사용 비용이 청구될 수 있습니다.

**Optional flags**:

* `--use_atomic_facts`: 이 flag 적용시, atomic fact generator를 사용하지 않고 미리 생성된 atomic facts를 가져와서 사용할 수 있습니다. 이 경우 `input_path`는 atomic facts를 포함하고 있는 입력 파일이어야 합니다. 제공된 `data/k_labeled/gen_bio-gpt_4-{kr/fr}-all.jsonl` 파일은 gpt-4로 생성된 atomic-facts를 포함하고 있습니다. (아래 실험 결과를 재현할 때 사용됩니다.)
* `--knowledge_source`: 기본적으로 kowiki-20240301.db로 설정되어 있습니다. (한국어 Wikipedia - 2024/03/01).

<!--개인 DB를 적용하고 싶다면 [아래 설정](#To-use-a-custom-knowledge-source) 방법을 참고하세요. 그리고 db 이름을 이 flag에 넣어주세요. -->

<br>
<br>

## 실험용 데이터

* 인물의 약력을 GPT-4로 생성한 문서들로 이루어진 데이터셋
    * 위치:
        * data/k_unlabeled/
            (atomic facts 불포함)
    * 파일이름:
        * gen_bio-gpt_4-kr-all.jsonl
        * gen_bio-gpt_4-fr-all.jsonl
</br>

    | 데이터 | 문서 수 |
    | --- | ---- |
    | 한국인 약력 | 64 |
    | 외국인 약력 | 50 |

<br>

## 성능 평가 결과

### 성능 평가용 데이터

* 실험용 데이터에 atomic facts 분할 결과를 추가한 데이터셋.
    Atomic facts 생성에 사용된 모델은 gpt-4-turbo 입니다.
    * 위치:
        * data/k_labeled/
            (atomic facts 포함)
    * 파일이름:
        * gen_bio-gpt_4-kr-all.jsonl
        * gen_bio-gpt_4-fr-all.jsonl

### 검증기에 사용된 모델

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

### System의 사실판단 성능 계산 방법

정답(ground truth)과 system의 결과 사이의 일치하는 정도를 계산하여 평가함
* 정답 파일위치 및 이름:
    * data/k_truth_annotations/gen_bio-gpt_4-kr-all-gold.jsonl
    * data/k_truth_annotations/gen_bio-gpt_4-fr-all-gold.jsonl

```bash
human_judge_data='data/k_truth_annotations/gen_bio-gpt_4-kr-all-gold.jsonl'
system_decisions_data='cache/gpt_4-kr-all/4_af_decisions_bm25_k5.jsonl'

python factscore/evaluate_system_vs_human_judgments ${human_judge_data} ${system_decisions_data}
```

* 검색기: BM25
</br>

----
\* 아래 실험 결과는 [KorFactScore 실행 방법](#korfactscore-실행-방법)을 통해 재현 가능합니다.
### 결과 (1) System의 사실판단 성능

* `System의 사실판단 성능`은 System 결과와 정답(ground truth)의 일치된 정도를 보여줍니다(단위: %).

| Model | Size | 한국인 <br> BM25 |  Cross-Encoder | 외국인 <br> BM25 |  Cross-Encoder |
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

\* `Cross-Encoder`는 ETRI 내부 기술로 개발된 검색기로서 기술이전을 통해 사용이 가능합니다.

### 결과 (2) 생성문서의 사실성 점수 (FactScore)

* `생성문서의 사실성 점수`는 입력된 문서의 원소 사실들(atomic facts)에 대해 system이 평가한 사실성 점수, 즉 FactScore입니다(단위: %).
    (Ground truth의 사실성 점수는 한국인 대상 46.8, 외국인 대상 80.1 입니다.)

| Model | Size | 한국인 <br> BM25 |  Cross-Encoder | 외국인 <br> BM25 |  Cross-Encoder |
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

<!-- 제외? ### 제한사항 
* 검증을 위한 지식베이스가 현재는 위키백과로 한정됨. 동일한 포맷의 지식베이스를 제공할 경우 확장 가능. -->

</br>
</br>
</br>

#### (이 패키지는 영어 생성 문서에 대한 원소 사실성 평가 기술인 [FActScore](https://github.com/shmsw25/FActScore)를 기반으로 개발되었습니다.)

</br>

## Citation 
이 패키지를 연구에 활용할 경우 아래와 같이 인용해주세요

* 노지현*, 김민호*, 배용진, 김현기, 이형직, 장명길, 배경만, "한국어 생성 문서의 원소 사실 관계에 대한 설명 기술," [Online] [https://github.com/ETRI-XAINLP/KorFactScore](https://github.com/ETRI-XAINLP/KorFactScore), 2024

</br>
</br>

**[Funding]** (4세부) 전문지식 대상 판단결과의 이유/근거를 설명가능한 전문가 의사결정 지원 인공지능 기술 개발
    * 사람중심인공지능핵심 원천기술개발 사업 > 사용자 맞춤형 플러그앤플레이 방식의 설명가능성 제공 기술 개발
        \(과학기술정보통신부 재원 \| 정보통신기획평가원 지원\)

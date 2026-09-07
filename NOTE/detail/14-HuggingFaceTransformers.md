# Hugging Face Transformers 기본 활용

> 출처 TIL: 260907

## 1. Hugging Face Hub

Hugging Face Hub는 모델, 데이터셋, 데모(Spaces)를 호스팅하는 git 기반 플랫폼입니다. GitHub과 비슷하지만 머신러닝 자산에 특화되어 있습니다. 각 모델은 하나의 git 저장소이고, 용량이 큰 가중치 파일은 git-LFS로 관리합니다.

모델 저장소의 표준 구성은 이렇습니다. config.json은 아키텍처와 하이퍼파라미터(레이어 수, hidden 차원, model_type 등)를 담습니다. model.safetensors(또는 예전 형식인 pytorch_model.bin)는 학습된 가중치입니다. tokenizer.json, vocab.txt, tokenizer_config.json 등은 토크나이저 파일입니다. README.md가 곧 Model Card입니다.

from_pretrained에 "조직명/모델명" 형태의 저장소 id를 넣으면 파일을 로컬 캐시(홈 디렉터리 아래 .cache/huggingface/hub)로 자동으로 내려받습니다. 접근 유형은 공개, gated(라이선스에 동의해야 받을 수 있음, 예를 들어 Llama), private로 나뉩니다. 저장소는 커밋 해시, 태그, 브랜치로 버전이 관리되며, revision 인자에 커밋 해시를 지정하면 특정 시점의 파일로 고정할 수 있습니다.

## 2. Hub 탐색 흐름

탐색은 대체로 검색어 입력, 필터링, 정렬, 모델 페이지 확인, Model Card 정독, Files 탭 확인, 저장소 id를 코드에 사용하는 순서로 진행됩니다.

필터로는 task(text-classification, text-generation 등), 라이브러리(transformers), 언어, 라이선스, 파라미터 크기를 걸 수 있습니다. 정렬은 trending, downloads, likes 기준이 있는데, downloads와 likes는 인기 지표일 뿐 품질이나 안전성을 보장하지는 않습니다. 모델 페이지에서는 Model Card 본문, 예제 코드 스니펫을 보여주는 "Use this model" 버튼, 실제 파일을 확인할 수 있는 Files and versions 탭, 이슈와 토론이 오가는 Community 탭을 볼 수 있습니다. 확인이 끝나면 저장소 id를 코드에 그대로 씁니다.

## 3. Model Card

Model Card는 README.md 파일이며, 상단의 YAML 프론트매터(메타데이터)와 본문(설명)으로 이루어집니다. 프론트매터에는 라이선스, 언어, 태그, 학습에 쓴 데이터셋, 평가 지표 등이 들어갑니다.

읽는 순서는 다음을 권합니다. 먼저 프론트매터와 태그에서 라이선스(상업적 사용이 가능한지), task, 언어를 확인합니다. 다음으로 모델 요약에서 무슨 모델인지, 베이스 모델은 무엇인지, 크기와 아키텍처를 봅니다. 그다음 의도된 용도와 범위 밖 용도를 보고 내 목적에 맞는지 판단합니다. 학습 데이터 항목에서 어떤 데이터로 배웠는지(도메인 편향을 파악), 평가 결과에서 벤치마크 점수와 비교 대상을 봅니다. 한계와 편향(Limitations & Bias) 항목에서 알려진 약점을 확인하고, 사용 예제 코드를 복사해 실행해 봅니다. 마지막으로 인용 형식과 라이선스 세부를 봅니다. 실무에서는 라이선스와 학습 데이터를 가장 먼저 보는 것이 좋습니다. 성능이 좋아도 라이선스가 맞지 않으면 쓸 수 없기 때문입니다.

## 4. pipeline() — 가장 쉬운 진입점

pipeline은 토크나이즈, 모델 추론, 후처리를 한 줄로 묶어 주는 고수준 API입니다. task 이름만 주면 그 task의 기본 모델을 자동으로 골라 내려받습니다.

    from transformers import pipeline

    clf = pipeline("sentiment-analysis")
    clf("이 영화 정말 최고였다")
    # [{'label': 'POSITIVE', 'score': 0.99}]

task 이름이 동작을 결정하며 sentiment-analysis, text-generation, ner, question-answering, summarization, fill-mask 등이 있습니다. model 인자로 특정 모델을 지정할 수 있고, 생략하면 task별 기본 모델을 씁니다. 입력은 문자열 하나 또는 리스트(배치)이고, 출력은 후처리까지 끝난 파이썬 객체(라벨과 점수, 생성된 문장 등)입니다. 내부적으로는 AutoTokenizer와 AutoModelFor 계열 클래스를 조합해 실행하며, pipeline은 그 위를 감싼 편의 계층입니다.

추상화 계층으로 보면, pipeline이 가장 쉽고 제어가 적으며, 그 아래에 토크나이저와 모델을 직접 다루는 AutoClass가 있고, 가장 아래에 커스텀 head와 학습 루프를 직접 짜는 완전 제어 단계가 있습니다.

## 5. AutoClass

AutoClass는 모델 아키텍처를 몰라도 알맞은 구현 클래스를 자동으로 골라 주는 팩토리입니다. config.json의 model_type이나 architectures 필드를 보고, BERT면 BertModel, GPT-2면 GPT2Model을 내부적으로 인스턴스화합니다.

주요 AutoClass로는 설정만 로드하는 AutoConfig, 토크나이저를 로드하는 AutoTokenizer, head 없는 베이스 모델을 로드하는 AutoModel, 베이스에 task head를 붙여 로드하는 AutoModelForSequenceClassification 같은 클래스들이 있습니다. 이 방식의 장점은 코드가 특정 모델에 묶이지 않아서, 모델을 바꿀 때 문자열 하나만 고치면 된다는 점입니다.

## 6. Base Model과 Task-specific Model

AutoModel로 로드하는 base model은 트랜스포머 본체만 있고, 출력은 hidden states, 즉 표현 벡터입니다. AutoModelForSequenceClassification처럼 AutoModelFor로 시작하는 task-specific model은 본체에 task head가 붙어 있고, 출력은 task logits(분류 점수, 어휘 점수 등)입니다.

여기에 함정이 하나 있습니다. base 체크포인트만 있는 저장소에서 AutoModelForSequenceClassification을 로드하면 head의 가중치가 없어서 무작위로 초기화되고, "일부 가중치가 초기화되지 않았으니 downstream task로 학습하라"는 경고가 뜹니다. 이 상태로 바로 추론하면 결과가 무의미하며, fine-tuning이 필요하다는 신호입니다.

## 7. AutoTokenizer와 AutoModel의 기본 출력

AutoTokenizer에 텍스트를 넣으면 딕셔너리가 나옵니다.

    tokenizer("안녕하세요", return_tensors="pt", padding=True, truncation=True)

    {
      'input_ids':      tensor([[101, 9521, ..., 102]]),
      'attention_mask': tensor([[1, 1, ..., 1, 0, 0]]),
      'token_type_ids': tensor([[0, 0, ..., 0]])
    }

input_ids는 토큰을 정수 id로 바꾼 것이고, attention_mask는 실제 토큰이 1, 패딩이 0으로 "패딩은 무시하라"는 표시입니다. token_type_ids는 문장 A와 B를 구분하며 BERT 계열에서만 나옵니다. return_tensors는 반환 형식(pt는 PyTorch 텐서), padding은 배치 안에서 길이를 맞추기 위한 채움, truncation은 너무 긴 입력 자르기입니다.

AutoModel(base model)의 출력은 BaseModelOutputWithPooling 형태입니다. last_hidden_state는 (batch, seq_len, hidden) 모양으로 마지막 레이어의 모든 토큰 표현이며 실무에서 주로 이것을 씁니다. pooler_output은 (batch, hidden) 모양으로 [CLS] 벡터를 선형층과 tanh에 통과시킨 것인데, BERT 계열에서만 나오고 신뢰도가 낮아 잘 쓰지 않습니다. hidden_states는 각 레이어별 출력의 튜플로 output_hidden_states를 켜야 나오고, attentions는 각 레이어의 어텐션 가중치로 output_attentions를 켜야 나옵니다.

## 8. last_hidden_state 해석

last_hidden_state의 모양은 (배치 크기, 시퀀스 길이, hidden 차원)입니다. [b, t, :] 벡터는 b번째 문장에서 t번째 토큰이 전체 문맥 속에서 갖는 의미, 즉 문맥이 반영된 임베딩입니다. 같은 단어라도 문장이 다르면 벡터가 달라집니다.

용도별로 쓰는 법이 다릅니다. 개체명 인식이나 품사 태깅 같은 토큰 분류는 각 위치의 벡터를 그대로 씁니다. BERT로 문장 임베딩을 얻을 때는 [CLS] 위치, 즉 시퀀스의 0번 위치 벡터를 쓰거나, 더 권장되는 방식으로 attention_mask를 이용해 패딩을 제외하고 토큰 벡터를 평균 내는 mean pooling을 씁니다. GPT 계열에서 다음 토큰을 예측할 때는 마지막 실제 토큰 위치의 벡터를 씁니다. 패딩 위치의 벡터는 의미가 없으므로 평균이나 풀링을 할 때 반드시 마스크로 제외해야 합니다.

## 9. Task Head

task head는 베이스 모델 위에 얹는 작은 신경망으로 보통 선형층 한두 개입니다. 표현(hidden)을 task 출력(logits)으로 바꿉니다.

Sequence Classification head는 pooled 표현(벡터 하나)을 받아 클래스 수만큼의 점수를 냅니다. Token Classification head는 각 위치의 벡터를 받아 위치마다 클래스 수만큼의 점수를 냅니다. Question Answering head는 각 위치의 벡터를 받아 위치마다 start와 end 두 점수를 냅니다. Language Modeling head는 각 위치의 벡터를 받아 위치마다 어휘 크기만큼의 점수를 내며, 보통 입력 토큰 임베딩과 가중치를 공유해 파라미터를 아낍니다. head는 downstream 데이터로 학습되는 부분입니다.

## 10. AutoModelForSequenceClassification과 AutoModelForCausalLM

| 구분 | ForSequenceClassification | ForCausalLM |
|---|---|---|
| 목적 | 문장이나 문서 분류 | 텍스트 생성 |
| 대표 모델 | BERT, RoBERTa | GPT-2, Llama, Qwen |
| head | 분류 head (pooled에서 num_labels로) | LM head (hidden에서 vocab으로) |
| 출력 logits 모양 | (batch, num_labels) | (batch, seq_len, vocab_size) |
| 로드 시 추가 인자 | num_labels, id2label, label2id | 보통 없음 |
| generate() | 불가 | 가능 |

## 11. classification logits와 vocabulary logits의 축

classification logits는 (batch, num_labels) 모양입니다. 0번 축은 배치 안의 각 샘플이고, 1번 축은 각 클래스입니다(예를 들어 부정은 0번, 긍정은 1번). 1번 축에서 argmax를 취하면 예측 클래스가 나오고, softmax를 취하면 확률이 나옵니다.

vocabulary logits는 (batch, seq_len, vocab_size) 모양입니다. 0번 축은 배치, 1번 축은 시퀀스 위치 t이며 각 위치는 "그 다음에 올 토큰"을 예측합니다. 2번 축은 어휘 사전 전체 토큰 각각의 점수로, 크기는 보통 5만에서 15만입니다. 마지막 축에서 argmax를 취하면 다음 토큰 id가 나옵니다. 학습할 때는 모든 위치의 logits를 다 쓰고(각 위치의 예측을 실제 다음 토큰과 비교), 생성할 때는 마지막 위치의 logits만 필요합니다.

## 12. 저장·재로드와 재현성 메타데이터

save_pretrained로 모델과 토크나이저를 각각 저장하고, from_pretrained로 다시 불러옵니다. 이때 모델과 토크나이저는 반드시 같은 경로에서 짝으로 로드해야 합니다. 토크나이저의 vocab과 모델의 임베딩 테이블 인덱스가 1대1로 대응하기 때문에, 짝이 어긋나면 오류 없이 조용히 틀린 결과가 나옵니다.

재현성을 위해 함께 기록해 둘 메타데이터가 있습니다. revision(커밋 해시)은 Hub 모델이 갱신될 수 있으므로 시점을 고정하는 데 씁니다. transformers와 torch 버전은 버전 간 동작 차이 때문에 기록합니다. torch_dtype(fp32, fp16, bf16)은 정밀도에 따라 출력이 미세하게 달라지므로 명시합니다. 랜덤 시드는 초기화와 샘플링을 재현하기 위해 고정합니다. safetensors 형식은 pickle의 보안 문제를 피하고 결정적으로 로드되므로 권장합니다. generation_config.json은 생성 파라미터(temperature, top_p 등)를 고정합니다. 학습을 한 경우에는 TrainingArguments와 데이터셋 버전까지 남겨야 학습 과정 전체를 재현할 수 있습니다.

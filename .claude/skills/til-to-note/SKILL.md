---
name: til-to-note
description: Incrementally turns this repo's TIL/*.md daily notes into topic-grouped study material under NOTE/. Content lives in four domain files — NOTE/1-수학.md, NOTE/2-머신러닝.md, NOTE/3-딥러닝.md, NOTE/4-LLM.md — each a fully-written prose explanation (no math formulas/LaTeX; English terms written as 한글(English), not phonetic transliteration) with a clickable table of contents. NOTE/summary.md is a single cross-domain keyword index that links into those files. Use this whenever the user asks to update, sync, regenerate, or build the NOTE folder from TIL, says things like "TIL 정리해줘", "노트 업데이트해줘", "교안 만들어줘/갱신해줘", "오늘 TIL을 노트에 반영해줘", or runs this skill with no arguments expecting a full incremental scan. Always check NOTE/.manifest.json and the existing NOTE files first — never regenerate everything from scratch; this skill's entire point is incremental, non-destructive updates.
---

# TIL → NOTE 교안 증분 생성

## 이 스킬이 하는 일과 하지 않는 일

TIL/*.md(하루치 학습 키워드 정리)를 네 개의 도메인으로 묶어 교안으로 재구성합니다.

- `NOTE/1-수학.md` · `NOTE/2-머신러닝.md` · `NOTE/3-딥러닝.md` · `NOTE/4-LLM.md` — 도메인별 상세 교안. 개념·배경·예시를 문단으로 풀어쓴 설명이며, 수식/LaTeX 없이 전부 말로 풀어씁니다. 이해를 돕는 Python 코드 스니펫은 유지 가능. 각 파일의 구조는 아래 "도메인 파일 구조" 참고.
- `NOTE/summary.md` — 네 도메인 전체를 한 파일에 담은 키워드 인덱스. 매일 위에서 아래로 훑어보며 리마인드하는 용도. `## 도메인` 아래에 `### 하위주제 · [상세 →](파일#anchor)` 블록들이 있고, 각 블록은 `- **키워드**: 한두 줄 설명` 불릿 목록입니다. 코드 없음.

네 도메인은 고정입니다. TIL 내용은 반드시 이 중 하나에 들어갑니다.

| 도메인 | 파일 | 범위 |
|---|---|---|
| 수학 | 1-수학.md | 선형대수·행렬분해, PCA의 수학적 기반 |
| 머신러닝 | 2-머신러닝.md | 지도·비지도 기초 모델, 앙상블, 편향-분산·규제, 평가지표·데이터 누수 |
| 딥러닝 | 3-딥러닝.md | 딥러닝 기초·PyTorch, CNN, RNN·LSTM, 트랜스포머 수학(어텐션 shape·미분·역전파) |
| LLM | 4-LLM.md | NLP 기초, 트랜스포머 아키텍처, 사전학습 LM, Hugging Face, 서빙·하네스 |

**절대 전체를 다시 쓰지 않습니다.** TIL은 매일 추가되는 데이터이므로, 이 스킬은 "지금까지 반영 안 된 부분만 찾아서 기존 문서에 이어 붙이거나, 바뀐 부분만 고치는" 증분 작업입니다. 이미 처리된 내용을 다시 훑어 전체 파일을 재작성하면, 사용자가 반영 이후 직접 다듬었을 수도 있는 부분을 조용히 덮어써버립니다 — 그래서 "무엇이 이미 반영됐는가"를 정확히 추적하는 것이 이 스킬의 핵심입니다.

## 도메인 파일 구조

```
# 딥러닝                                  ← 도메인 제목 (h1)

<한 줄 소개>

> 출처 TIL: 260810, 260819, ...           ← 이 도메인에 반영된 모든 TIL 날짜

## 목차                                   ← regen_toc.py 로 자동 생성 (직접 손대지 않음)

- [딥러닝 기초와 PyTorch](#딥러닝-기초와-pytorch)
    - [1. 머신러닝과 딥러닝의 접근 차이](#1-머신러닝과-딥러닝의-접근-차이)
    ...

---

## 딥러닝 기초와 PyTorch                   ← 하위주제 (h2). 옛 "그룹" 하나에 해당

> 출처 TIL: 260819, 260820, ...           ← 이 하위주제의 TIL 날짜

### 1. 머신러닝과 딥러닝의 접근 차이        ← 절 (h3). 하나의 개념

<문단 설명>

### 2. ...
```

- 도메인 파일은 여러 개의 `## 하위주제`를 담고, 각 하위주제는 여러 개의 `### N. 절`을 담습니다. 절 번호는 하위주제마다 1부터 새로 매깁니다.
- 도메인 파일 맨 위의 `> 출처 TIL`은 그 도메인의 모든 TIL을 합집합으로 나열하고, 각 `## 하위주제` 아래의 `> 출처 TIL`은 그 하위주제 것만 나열합니다.
- 절 사이에 다른 곳을 참고하면 좋은 내용이 있으면 절 제목 바로 아래 한 줄로 표시합니다:
  `> 참고 · <설명>은 [딥러닝 · 54. Batch Normalization](3-딥러닝.md#54-batch-normalization)`
  같은 파일 안이면 파일명 없이 `(#anchor)`, 다른 도메인 파일이면 `(N-도메인.md#anchor)`.

## 왜 매니페스트가 필요한가

TIL 파일이 "새로 추가됐는지" 또는 "내용이 바뀌었는지"를 매번 눈으로 비교하는 것은 비용이 크고 실수하기 쉽습니다. 그래서 `NOTE/.manifest.json`에 각 TIL 파일의 마지막 반영 시점 해시와, 그 내용이 어느 도메인으로 들어갔는지(`group` = `1-수학` / `2-머신러닝` / `3-딥러닝` / `4-LLM`)를 기록해둡니다. "해시가 같다 = 이 TIL은 이미 완전히 반영됨"을 기계적으로 판단합니다. 이 비교는 `scripts/til_manifest.py`가 처리합니다 — "어느 도메인·하위주제에 속하는가", "어떻게 풀어 쓸 것인가"는 모델이 판단합니다.

## 절차

### 1. 스캔

```bash
python .claude/skills/til-to-note/scripts/til_manifest.py scan
```

`{new: [...], changed: [...], unchanged: [...], missing: [...]}` 가 출력됩니다.

- `new`와 `changed`가 모두 비어 있으면 **할 일이 없는 것**입니다. "NOTE는 이미 최신 상태입니다"라고 보고하고 끝냅니다. (여기서 멈추지 않고 아무튼 뭔가 다시 쓰는 것이 이 스킬이 저지르기 가장 쉬운 실수입니다.)
- `changed`의 각 항목은 이전에 반영됐던 도메인(`previous_group`)을 우선 검토 대상으로 삼습니다.
- `missing`은 매니페스트엔 있는데 실제 파일이 없는 것입니다. 사용자에게 알리되 NOTE에서 자동으로 지우지는 않습니다.

### 2. 기존 NOTE를 먼저 읽는다

처리할 TIL이 하나라도 있으면, 반영 위치를 판단하기 전에 **먼저** 대상 도메인 파일의 `## 목차`(하위주제·절 목록)와 `NOTE/summary.md`의 해당 `## 도메인` 섹션을 훑어봅니다. 이렇게 해야 어느 하위주제에 붙일지, 이미 있는 절과 겹치지 않는지, 기존 문체를 어떻게 잇는지 알 수 있습니다.

### 3. 새/변경된 TIL마다 분류하고 반영

각 TIL 파일을 전체 읽고 다룬 주제를 봅니다. 네 도메인 중 하나로 분류합니다(위 표 기준). 한 TIL이 여러 도메인에 걸치면 절 단위로 쪼개 각 도메인에 나눠 반영합니다.

**기존 하위주제에 붙는 경우**: 도메인 파일에서 그 `## 하위주제`를 찾아, TIL의 각 절을 `### N. 제목`으로 이어 붙입니다(절 번호는 그 하위주제의 마지막 번호 다음). `NOTE/summary.md`에서 같은 `### 하위주제 · [상세 →]` 블록을 찾아 불릿을 이어 붙입니다. 두 곳의 `> 출처 TIL`에 새 날짜를 추가합니다. 기존 절·불릿은 건드리지 않습니다.

**새 하위주제가 필요한 경우**(그 도메인에 아직 없는 주제): 도메인 파일에서 내용상 자연스러운 위치(보통 관련 하위주제 뒤, 없으면 파일 끝)에 `## 새 하위주제` + `> 출처 TIL` + `### 1. ...` 절들을 추가합니다. `NOTE/summary.md`의 해당 `## 도메인` 섹션 안, 순서상 맞는 자리에 `### 새 하위주제 · [상세 →](N-도메인.md#anchor)` 블록을 추가합니다(anchor는 4단계에서 regen_toc.py 실행 후 `## 목차`에서 그대로 복사).

**TIL이 바뀐 경우(`changed`)**: 이전 버전과 비교할 방법이 없으므로(해시만 저장), 현재 TIL 전체를 기준으로 도메인 파일에서 제목이 같거나 매우 유사한 `###` 절과 summary.md의 대응 불릿을 찾아 그 부분만 고칩니다. 못 찾으면 새로 추가합니다. 절대 파일 전체를 지우고 새로 쓰지 않습니다.

**다섯 번째 도메인이 필요해 보이는 경우**: 네 도메인 중 어디에도 안 맞는 내용은 드뭅니다. 대화형 세션이면 새 도메인 파일을 만들지 사용자에게 짧게 물어봅니다. 자동 실행이면 가장 가까운 도메인에 새 하위주제로 넣고, 결과 보고에 "이 내용은 도메인 X에 임시로 넣었으니 확인 바람"이라고 남깁니다.

### 4. 목차 재생성

절을 추가·수정·이동한 도메인 파일마다 목차를 다시 만듭니다(anchor는 GitHub 규칙을 정확히 따라야 하므로 직접 쓰지 않습니다):

```bash
python .claude/skills/til-to-note/scripts/regen_toc.py NOTE/3-딥러닝.md
```

여러 파일을 손댔으면 다 나열합니다. 마지막에 `--check`로 전체 확인:

```bash
python .claude/skills/til-to-note/scripts/regen_toc.py --check NOTE/1-수학.md NOTE/2-머신러닝.md NOTE/3-딥러닝.md NOTE/4-LLM.md
```

`## 목차`의 anchor와 `NOTE/summary.md`의 `[상세 →]` anchor, 그리고 `> 참고` 링크는 모두 이 규칙으로 나온 값이어야 합니다.

### 5. 작성 규칙

- **수식/LaTeX 금지**: 공식을 그대로 옮기지 말고 전부 말로 풀어씁니다. `det(A) = a×d - b×c` → "행렬식은 대각선 방향 값들의 곱에서 반대 대각선 방향 값들의 곱을 뺀 값입니다".
- **코드**: 도메인 파일에는 이해를 돕는 Python 코드 스니펫(펜스 코드 블록)을 유지해도 됩니다. summary.md에는 코드를 넣지 않습니다.
- **summary.md 스타일**: `- **키워드**: 한두 줄 설명` 불릿. 상세본 없이도 그 자체로 훑을 수 있게 씁니다. `## 도메인` 순서(수학 → 머신러닝 → 딥러닝 → LLM)와 그 안의 `### 하위주제` 순서는 도메인 파일의 하위주제 순서와 같게 유지합니다.
- **도메인 파일 스타일**: `###` 절마다 하나의 개념. 배경·정의·예시·실무 함의를 문단으로. 기존 문체(존댓말, 문단 중심)를 따릅니다.
- **용어 표기**: 영어 용어를 소리 나는 대로 한글로 옮기지 않습니다("베스트 프랙티스" → "모범 사례(best practice)"). 첫 등장 시 `한글 용어(English)` 로 병기하고 이후에는 한글만 씁니다. 굳어진 차용어(프롬프트·모델·토큰)와 약어(LLM·PCA·LoRA), 코드 식별자는 예외. TIL 서식 규칙 12와 같은 원칙입니다.
- **출처 표기**: 도메인 파일 맨 위 `> 출처 TIL`(도메인 전체 합집합), 각 `## 하위주제` 아래 `> 출처 TIL`(그 하위주제 것), `NOTE/summary.md`의 각 `### 하위주제` 블록 아래 `> 출처 TIL` — 세 곳을 새 날짜가 들어올 때마다 갱신합니다.

### 6. 매니페스트 갱신

반영을 마친 TIL마다:

```bash
python .claude/skills/til-to-note/scripts/til_manifest.py update 260815.md 2-머신러닝
```

`group`은 `1-수학` / `2-머신러닝` / `3-딥러닝` / `4-LLM` 중 하나입니다. 한 TIL을 여러 도메인에 나눠 반영했으면 가장 비중이 큰 도메인을 기록하고, 결과 보고에 "이 TIL은 N개 도메인에 나눠 반영됨"이라고 남깁니다.

### 7. 결과 보고

무엇을 어디에 반영했는지(어떤 TIL → 어떤 도메인 → 어떤 하위주제/절), 새 하위주제를 만들었는지, `missing` 항목이 있었는지를 짧게 보고합니다.

## 형식만 바뀐 TIL 처리

`til-format` 스킬로 TIL 파일의 서식만 고친 경우(내용 불변), NOTE 재동기화는 불필요합니다. 해시만 어긋나 다음 `scan`에서 `changed`로 잡히므로, 해당 TIL의 현재 도메인으로 매니페스트 해시만 갱신합니다:

```bash
python .claude/skills/til-to-note/scripts/til_manifest.py update 260904.md 4-LLM
```

## 예시

**입력 상황**: `TIL/260910.md` 추가, 내용은 "LoRA와 QLoRA" 한 절.

1. `scan` → `new: ["260910.md"]`
2. LoRA는 파인튜닝 기법 → LLM 도메인. `NOTE/4-LLM.md`의 `## 목차`를 보니 "사전학습 언어모델" 하위주제는 있지만 파인튜닝 전용 하위주제는 없음.
3. `NOTE/4-LLM.md`의 "사전학습 언어모델" 뒤에 `## 파라미터 효율적 파인튜닝` + `> 출처 TIL: 260910` + `### 1. LoRA` `### 2. QLoRA` 추가. `NOTE/summary.md`의 `## LLM` 안 해당 위치에 `### 파라미터 효율적 파인튜닝 · [상세 →](4-LLM.md#파라미터-효율적-파인튜닝)` 블록 + 불릿 추가.
4. `python .claude/skills/til-to-note/scripts/regen_toc.py NOTE/4-LLM.md` → 목차의 새 anchor 확인, summary.md의 `[상세 →]` anchor를 그 값으로 맞춤.
5. `til_manifest.py update 260910.md 4-LLM`
6. "260910.md를 LLM 도메인의 새 하위주제 '파라미터 효율적 파인튜닝'으로 반영했습니다"라고 보고.

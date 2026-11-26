"""main.py의 todo(...) 부분을 작성하세요. support.py와 checks.py는 그대로 사용합니다."""
import copy
import hashlib
import json
import sys

import support


def todo(message):
    raise NotImplementedError(message)


# 필수 1. 필요한 자료와 사용자 기억 관리하기
def select_context(items, user_id, now):
    kept = []
    for item in items:
        # allowed = todo("접근 범위, 신뢰, 현행 여부와 만료 조건")
        allowed = (
            item["user_id"] in ("public", user_id)
            and item["trusted"] is True
            and item["active"] is True
            and (item["expires_at"] is None or item["expires_at"] > now)
        )
         
        if allowed:
            kept.append(copy.deepcopy(item))

    # 이유: 높은 우선순위의 자료를 먼저 고르면 어떤 문제가 생기는지 적으세요. 
    # 우선순위부터 보고 고르면 높은 숫자를 갖고 있는 자료가 맨 앞자리를 차지합니다.
    return sorted(kept, key=lambda item: (-item["priority"], item["id"]))


def memory_action(records, user_id, action, name, value=None):
    support.validate_memory_input(user_id, action, name, value)
    key = f"{user_id}:{name}"
    if action == "read":
        #return todo("현재 사용자의 기억 조회")
        return copy.deepcopy(records.get(key))

    if action == "create":
        if key in records:
            raise ValueError("이미 있는 기억입니다. update를 사용하세요.")
        # records[key] = todo("새 기억 저장")
        records[key] = copy.deepcopy(value)

    elif action == "update":
        if key not in records:
            raise KeyError("수정할 기억이 없습니다.")
        #records[key] = todo("기존 기억 수정")
        records[key] = copy.deepcopy(value)
        
    elif action == "delete":
        #return todo("현재 사용자의 기억 삭제")
        return records.pop(key, None)

    return copy.deepcopy(records[key])


def build_messages(prefix, question, context, turns, summary):
    data = json.dumps({"reference_data": context, "summary": summary}, ensure_ascii=False)
    history = [message for turn in turns for message in turn["messages"]]
    # 이유: 장기 기억과 세션 상태의 차이, 자료와 지시를 나누는 이유를 적으세요.
    # 장기 기억은 세션이 끝나도 파일에 남고 세션 상태는 이번 대화에서만 쓰고 사라집니다. 
    # 자료를 지시(developer)와 나눠 user 데이터로 두어야 본문 속 명령문이 서비스 지시가 되지 않도록 합니다.

    #return todo("고정 지시, 참고 자료, 대화, 현재 질문 순서의 메시지")
    return [
         {"role": "developer", "content": prefix},
         {"role": "user", "content": data},
         *history,
         {"role": "user", "content": question},
     ]


# 필수 2. 긴 대화를 줄이고 반복 입력 재사용하기
def summarize(turns):
    summary = {}
    for turn in turns:
        for name, value in support.fact_lines(turn["messages"][0]["content"]):
            # summary[name] = todo("최신 값과 출처 턴")
            summary[name] = {"value": value, "source_turn_id": turn["id"]}
    return summary


def select_window(turns, limit, reserve, margin, make_messages, count):
    support.validate_budget(limit, reserve, margin)
    # budget = todo("입력에 쓸 수 있는 예산")
    budget = limit - reserve - margin

    if budget <= 0 or count(make_messages([])) > budget:
        raise ValueError("고정 입력만으로 예산을 넘었습니다.")
    kept = []
    for turn in reversed(turns):
        candidate = [turn, *kept]
        #if todo("전체 후보 입력이 예산을 넘는지"):
        if count(make_messages(candidate)) > budget:
            break
        kept = candidate
    # 관측: 압축 비교 후 토큰 수, 빠진 사실, 선택한 한도를 기록합니다.
    # (압축전) 1938, (최근 대화) 1699, (요약+최근 대화) 1695 
    # 최근만 남기는 방식은 담당자·마감일·승인조건이 빠진다. 
    # 선택한 한도는 2300, 방식은 요약+최근 대화. 요약에 세 사실이 남아 전체 대화와 같은 답이 나온다


    return kept


def cache_key(prefix, user_id):
    #return todo("사용자와 고정 지시로 만드는 캐시 키")
    return hashlib.sha256(f"{user_id}\n{prefix}".encode("utf-8")).hexdigest()
    # 관측: 실제 캐시 읽기/쓰기와 비용을 확인하고 미관측이면 그대로 적습니다.
    # (캐시 비사용) 쓰기 0, 읽기 0 / (첫 사용) 쓰기 1228 / (재사용) 읽기 1228
    # 입력 비용은 비사용 0.000307, 첫 사용 0.000368, 재사용 0.000086달러. 쓰기는 더 비싸고 읽기부터 싸진다.
    # 오염 자료 tainted-document, unverified-tool은 최종 입력에 없다.

# 선택 심화 1. 새 대화만 반영하고 변경 이력 남기기
def merge_summary(previous, turns):
    return todo("선택 심화: 새 턴의 요약과 변경 이력")


if __name__ == "__main__":
    raise SystemExit(support.cli(sys.modules[__name__]))

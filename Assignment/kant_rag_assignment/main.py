"""KANT NEXT DevDesk RAG. 이 파일의 빈칸만 작성하세요."""
from qdrant_client import models

import support


def todo(message):
    raise NotImplementedError(message)


# 필수 1. 문서를 저장하고 현행 자료만 검색하기
def prepare_chunks(documents):
    chunks = []
    for document in documents:
        seen = set()
        number = 0
        for paragraph in document["paragraphs"]:
            #text = todo("문단의 연속 공백을 한 칸으로 정리하세요.")
            text = " ".join(paragraph.split())

            if not text or text == "KANT NEXT | 내부 문서" or text in seen:
                continue
            seen.add(text)
            #chunk = todo("문서 정보 5개와 chunk_id, text를 담은 사전을 만드세요.")

            chunk = {
                 "document_id": document["document_id"],
                 "title": document["title"],
                 "category": document["category"],
                 "status": document["status"],
                 "source": document["source"],
                 "chunk_id": f"{document['document_id']}:{number}",
                 "text": text,
            }

            chunks.append(chunk)
            number += 1
    # 이유: 문단을 유지하는 이유와 보존해야 할 메타데이터를 한두 줄로 적으세요.
    # 더 나누면 근거 문장이 잘리므로 문단 단위를 유지한다
    # status, category, chunk_id, document_id, source, title

    return chunks


def build_index(chunks, embedder):
    texts = ["passage: " + row["title"] + "\n" + row["text"] for row in chunks]
    #vectors = todo("texts를 정규화된 문서 벡터로 바꾸세요.")
    vectors = embedder.encode(texts, normalize_embeddings=True)

    client = support.empty_store(len(vectors[0]))
    points = []
    for i, (row, vector) in enumerate(zip(chunks, vectors, strict=True)):
        #point = todo("i, vector, row를 Qdrant PointStruct에 연결하세요.")
        point = models.PointStruct(id=i, vector=[float(value) for value in vector], payload=row)
        points.append(point)
    client.upsert(collection_name="devdesk", points=points, wait=True)
    return client


def search(client, embedder, question, category, top_k=2):
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise ValueError("top_k는 1~5의 정수여야 합니다.")
    vector = embedder.encode(["query: " + question], normalize_embeddings=True)[0]
    #scope = todo("현행 문서이면서 지정한 category인 Qdrant 필터를 만드세요.")
    scope = models.Filter(must=[
            models.FieldCondition(key="status", match=models.MatchValue(value="current")),
            models.FieldCondition(key="category", match=models.MatchValue(value=category)),
    ])
    
    hits = client.query_points(collection_name="devdesk", query=list(vector),
                               query_filter=scope, limit=top_k, with_payload=True).points
    return [{**hit.payload, "score": hit.score} for hit in hits]


# 필수 2. 별칭 검색을 개선하고 답변을 비교하기
def rewrite_query(question, aliases):
    rewritten = question
    for alias, formal_name in aliases.items():
        #rewritten = todo("현재 문자열에서 alias를 formal_name으로 바꾸세요.")
        rewritten = rewritten.replace(alias, formal_name)
    return rewritten


def answer_question(question, rows, generate):
    if not rows:
        return {"answer": "확인할 문서가 없습니다.", "citations": [], "needs_human": True}
    #context = todo("각 검색 결과를 [chunk_id] 본문으로 묶고 빈 줄로 연결하세요.")
    context = "\n\n".join(f"[{row['chunk_id']}] {row['text']}" for row in rows)

    reply = support.validate_shape(generate(question, context))
    allowed = {row["chunk_id"] for row in rows}
    #if todo("검색 결과에 없는 출처가 하나라도 있는지 검사하세요."):
    if any(citation not in allowed for citation in reply["citations"]):
        raise ValueError("검색하지 않은 출처가 있습니다.")
    
    if reply["needs_human"] == bool(reply["citations"]):
        raise ValueError("답변에는 출처가 필요하며 담당자 확인에는 출처를 비웁니다.")
    return reply


def make_golden():
    #return todo("답이 있는 사례와 자료가 없는 사례를 하나씩 작성하세요.")
    return [
         {
             "question_id": "mine-vpn-disconnect",
             "question": "초록문 연결이 끊기면 어떻게 해야 하나요?",
             "category": "access",
             "reference_answer": "먼저 네트워크 상태를 확인하고, 문제가 계속되면 오류 시각과 화면을 개발지원팀에 전달합니다.",
             "relevant_document_ids": ["vpn"],
             "required_terms": ["네트워크 상태", "개발지원팀"],
         },
         {
             "question_id": "mine-overseas-lodging",
             "question": "해외 출장 숙박비 한도는 얼마인가요?",
             "category": "overseas",
             "reference_answer": "제공 문서에는 해외 출장 숙박비 규정이 없어 담당자 확인이 필요합니다.",
             "relevant_document_ids": [],
             "required_terms": [],
         },
     ]


def measure(case, rows, reply):
    expected = set(case["relevant_document_ids"])
    found = {row["document_id"] for row in rows}
    #recall = todo("필요한 문서 중 찾은 비율을 계산하세요. 필요한 문서가 없으면 None입니다.")
    recall = len(expected & found) / len(expected) if expected else None
    answer_ok = (all(term in reply["answer"] for term in case["required_terms"])
                 and not reply["needs_human"]) if expected else reply["needs_human"]
    failure = "없음"
    if expected and recall < 1:
        failure = "검색 누락"
    elif not answer_ok:
        failure = "답변 확인 필요" if expected else "보류 실패"
    # 관측: 실제 evaluate 결과에서 개선, 악화 질문과 답변 본문을 확인해 기록합니다.
    # Recall과 answer_check는 변환 전후 모두 1.0으로 개선·악화된 질문이 없습니다.

    return {"document_recall": recall, "answer_check": bool(answer_ok), "failure": failure}


# 선택 심화 1. 두 검색 순위를 합쳐 근거를 모으기
def fuse_rankings(rankings, top_k=2):
    # 선택 심화: 문제의 RRF 규칙에 맞춰 직접 작성하세요.
    return todo("검색 순위 목록을 RRF 점수로 합치세요.")


if __name__ == "__main__":
    support.cli(globals())

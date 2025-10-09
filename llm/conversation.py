import time
from app.session import Session
from utils.utils import dprint, lprint
import re 
import asyncio
from llm.openai import chorok_answer, hodol_greeting
import orjson as json
def jdumps(o): return json.dumps(o).decode()

SILENCE_PATTERN = re.compile(r"<\s*silence\s+(\d+(?:\.\d+)?)\s*>", re.IGNORECASE)

def split_by_silence_markers(text: str):
    """
    '<silence N>' 기준으로 텍스트/무음 명령을 순서대로 반환.
    반환 예) ["Hello.", ("__silence__", 3.0), "How are you?"]
    """
    parts = []
    pos = 0
    for m in SILENCE_PATTERN.finditer(text):
        if m.start() > pos:
            seg = text[pos:m.start()].strip()
            if seg:
                parts.append(seg)
        dur = float(m.group(1))
        parts.append(("__silence__", dur))
        pos = m.end()
    # 꼬리 텍스트
    tail_seg = text[pos:].strip()
    if tail_seg:
        parts.append(tail_seg)
    return parts


def reset_conversation(sess: Session):
    sess.transcripts.append(sess.current_transcript)
    sess.current_transcript = ""
    sess.outputs.append(sess.answer)

async def conversation_worker(sess: Session):
    while sess.running:
        text = await sess.answer_q.get()
        await answer_one(sess, text)

async def answer_one(sess: Session, transcript: str):
    st = time.time()
    sess.current_transcript += " " + transcript
    answer_text = await run_answer_async(sess)  # 내부에서 run_in_executor 사용
    dprint(f"[Answer {time.time() - st:.2f}s] - {answer_text!r}")
    sess.answer = answer_text.strip()

    sess.transcripts.append(sess.current_transcript)
    sess.current_transcript = ""
    sess.outputs.append(sess.answer)
    
    # 여기서 '비동기적으로' 추가 작업 실행 (await 하지 않음)
    # asyncio.create_task(postprocess_after_answer(sess))

# async def postprocess_after_answer(sess: Session):
#     try:
#         await summarize_memory(sess)
#     except Exception as e:
#         lprint(f"[postprocess_after_answer] error: {e}")

# async def summarize_memory(sess):
#     conversations = """
# """
#     update_memory(
#         conversations=conversations,
#         original_memory=sess.

async def answer_greeting(sess: Session):
    loop = asyncio.get_running_loop()

    def run_blocking():
        return hodol_greeting(
            language=sess.language,
            name=sess.name,
            current_time=sess.current_time
        )

    output = await loop.run_in_executor(None, run_blocking)
    answer_text = (output.get("text", "") or "").strip()

    if answer_text:
        loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, answer_text)
        loop.call_soon_threadsafe(
            sess.out_q.put_nowait,
            jdumps({
                "type": "translated",
                "script": sess.current_transcript,
                "text": answer_text,
                "is_final": True
            })
        )

async def run_answer_async(sess: Session) -> str:
    loop = asyncio.get_running_loop()
    sent_chars = 0
    
    def safe_push_tts(text: str):
        def _f():
            sess.tts_in_q.put_nowait(text)  # 절대 await 안 함
        loop.call_soon_threadsafe(_f)

    def safe_push_out(msg: dict):
        """
        send translated text to client
        """
        def _f():
            sess.out_q.put_nowait(jdumps(msg))
        loop.call_soon_threadsafe(_f)

    def clean_text(text: str) -> str:
        return re.sub(r"<[^>]*>", "", text)

    def on_token(tok: str):
        safe_push_out({"type": "translated", "text": tok, "is_final": False})
        return

    def run_blocking():
        return chorok_answer(
            prev_scripts=sess.transcripts[-6:],
            prev_answers=sess.outputs[-6:],
            input=sess.current_transcript,
            language=sess.language,
            onToken=on_token,
            current_time=sess.current_time
        )
    
    output = await loop.run_in_executor(None, run_blocking)
    answer_text = output.get("text", "") or ""
    
    pieces = split_by_silence_markers(answer_text)
    
    def push_piece(piece):
        if isinstance(piece, tuple) and len(piece) == 2 and piece[0] == "__silence__":
            # 무음 컨트롤 메시지 (튜플) - 대기열에 그대로 넣음
            sess.tts_in_q.put_nowait(("__silence__", piece[1]))
        else:
            # 일반 텍스트
            sess.tts_in_q.put_nowait(piece)
    
    # 순서를 유지한 채 한 번에 밀어넣기 (await 금지, 콜백으로 넣음)
    def _push_all():
        for p in pieces:
            push_piece(p)
    
    loop.call_soon_threadsafe(_push_all)
    
    await asyncio.sleep(0)
    cts = sess.current_transcript
    
    # 최종 알림 (그대로 둠)
    def _g():
        sess.out_q.put_nowait(jdumps({"type": "translated", "script": cts, "text": answer_text, "is_final": True}))
    loop.call_soon_threadsafe(_g)
    
    return answer_text


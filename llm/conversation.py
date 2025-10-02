import time
from app.session import Session
from utils.utils import dprint, lprint
import re 
import asyncio
from llm.openai import chorok_answer, hodol_greeting
import orjson as json
def jdumps(o): return json.dumps(o).decode()

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
            name=sess.name
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
        )

    output = await loop.run_in_executor(None, run_blocking)
    answer_text = output.get("text", "") or ""

    # 남은 꼬리 한 번만 푸시 (여기도 await 쓰지 않는 게 포인트)
    tail = clean_text(answer_text[sent_chars:]).strip()
    
    if len(tail) > 2:
        def _f():
            sess.end_translation_time = time.time()%1000
            sess.tts_in_q.put_nowait(tail)
        loop.call_soon_threadsafe(_f)

    await asyncio.sleep(0)
    cts = sess.current_transcript
    
    # 최종 알림(중복 방지)
    def _g():
        sess.out_q.put_nowait(jdumps({"type": "translated", "script": cts, "text": answer_text, "is_final": True}))
    loop.call_soon_threadsafe(_g)

    return answer_text

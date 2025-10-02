# --- MUST be at the very top (before any vllm/torch import) ---
import queue
import os, socket, multiprocessing as mp
import asyncio
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import orjson as json
import os
import re
import time
from websockets.asyncio.client import connect as ws_connect
import numpy as np
import librosa
import torch
import os, multiprocessing as mp
from librosa.util import normalize
import threading
import inspect

from stt.asr import load_asr_backend
from stt.vad import check_audio_state
from utils.process import process_data_to_audio
from app.session import Session, arm_end_timer, cancel_end_timer
from app.session_control import teardown_session, outbound_sender
from utils.text_process import text_pr
from utils.process import get_volume, pcm16_b64
from utils.utils import dprint, lprint
from llm.translate import translator_worker

from concurrent.futures import ThreadPoolExecutor

ENC_EXEC = ThreadPoolExecutor(max_workers=2)

DEFAULT_VOICE_PATH = "./output.wav"
INPUT_SAMPLE_RATE = 24000
WHISPER_SR = 16000
END_WORDS = [".", "!", "?", "。", "！", "？"]

filler_audios = []

def jdumps(o): return json.dumps(o).decode()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI()

global ASR

ASR = None
LLM = None

sessions: Dict[int, Session] = {}  # id(ws)로 매핑

@app.on_event("startup")
def init_models():
    global filler_audios
    # 멀티프로세싱/환경 변수는 가장 먼저
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    # 파이썬 멀티프로세싱 start method도 spawn으로
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    filler_audios_path = []
    for p in filler_audios_path:
        audiod, sr = librosa.load(p, sr=24000, mono=True)
        audiod = torch.tensor(audiod)
        filler_audios.append(pcm16_b64(audiod))

    global LLM

async def transcribe_pcm_generic(audios, sample_rate: int, channels: int, language: str) -> str:
    if not audios:
        return ""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: ASR.transcribe_pcm(audios, sample_rate, channels, language=language)
    )

async def stt_worker(sess: Session, in_q: asyncio.Queue, out_q: asyncio.Queue):
    try:
        while True:
            pcm_bytes = await in_q.get()  # 메인 루프와 독립적으로 대기/실행
            try:
                # 기존 함수 재사용 (여기서 await 해도 메인 루프는 안 멈춤)
                sttstart = time.time()
                text = await transcribe_pcm_generic(
                    audios=pcm_bytes,
                    sample_rate=16000,
                    channels=sess.input_channels,
                    language=sess.in_language
                )
                dprint(f"[stt_worker] {time.time() - sttstart:.4f}s")
                await out_q.put({"type": "delta", "text": text})
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[stt_worker] error: {e}")
            finally:
                in_q.task_done()
    finally:
        # drain 방지용
        while not in_q.empty():
            try:
                in_q.get_nowait(); in_q.task_done()
            except Exception:
                break

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session(input_sr=INPUT_SAMPLE_RATE, input_channels=1)
    sessions[id(ws)] = sess

    sess.sender_task = asyncio.create_task(outbound_sender(sess, ws))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(jdumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                t = data.get("type")

                # (A) 핑-퐁: 시계 오프셋 추정용
                if t == "latency.ping":
                    t1 = int(time.time() * 1000)
                    t2 = int(time.time() * 1000)
                    await ws.send_text(jdumps({
                        "type": "latency.pong",
                        "t0": data["t0"], "t1": t1, "t2": t2
                    }))
                    if not sess.is_network_logging:
                        sess.is_network_logging = True
                        lprint("network latency logging started")
                    continue
                if t == 'scriptsession.setvoice':
                    aud = data.get("audio")
                    lprint("Got ref voice!")
                    if aud:
                        sess.ref_audios = queue.Queue()
                        audio = process_data_to_audio(aud, input_sample_rate=24000, whisper_sr=WHISPER_SR)
                        if audio is None:
                            dprint("[NO AUDIO]")
                            continue
                        sess.ref_audios.put(audio)
                        continue

                # 1) 세션 시작: OpenAI Realtime WS 연결
                if t == "scriptsession.start":
                    global ASR
                    lprint("Start ", data);

                    if sess.in_language != data.get("in_language", "ko") or ASR is None:
                        sess.in_language = data.get("in_language", "ko")
                        ASR = load_asr_backend(kind=sess.in_language)

                    sess.out_language = data.get("out_language", "en")
                    
                    if sess.stt_task is None:
                        sess.stt_task  = asyncio.create_task(stt_worker(sess, sess.stt_in_q, sess.stt_out_q))
                    
                    if sess.stt_out_consumer_task is None:
                        sess.stt_out_consumer_task = asyncio.create_task(stt_out_consumer(sess))

                # 2) 오디오 append → Open AI로 그대로 전달
                elif t == "input_audio_buffer.append":
                    try:
                        aud = data.get("audio")
                        if aud:
                            # input audio sample rate -> 필요한 sample rate로 변환 = 16000
                            # 현재 한번 입력 단위는 80ms
                            audio = process_data_to_audio(aud, input_sample_rate=INPUT_SAMPLE_RATE, whisper_sr=WHISPER_SR)
                            if audio is None:
                                dprint("[NO AUDIO]")
                                continue

                            # === 여기서 VAD 검사 ===
                            vad_event = check_audio_state(audio)
                            audio_duration = audio.shape[-1] / WHISPER_SR
                            # 여기서 몇 ms 단위로 계속해서 찍히는지 측정 # 1~2ms 정도는 이유없이 차이날 수도 있다.
                            # print(f"[{time.time()%1000}] VAD event: [{vad_event}] ", audio.shape)

                            if sess.current_audio_state != "start":
                                sess.pre_roll.append(audio)

                                if vad_event == "start":
                                    cancel_end_timer(sess)
                                    # print("Come", get_volume(np.concatenate(list(sess.pre_roll) + [audio])))
                                    if not get_volume(np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False))[1] > 0.02:
                                        continue
                                    sess.current_audio_state = "start"
                                    if len(sess.pre_roll) > 0:
                                        sess.audios = np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False)
                                    else:
                                        sess.audios = audio.astype(np.float32, copy=False)
                                    print("[Voice Start] ", get_volume(sess.audios))
                                    sess.pre_roll.clear()
                                    sess.buf_count = 0
                                # 아직 start가 아니면(=무음 지속) 계속 프리롤만 업데이트하고 다음 루프
                                continue
                            
                            sess.audios = np.concatenate([sess.audios, audio])
                            sess.buf_count += 1

                            # 중간에 끊기
                            # if len(sess.transcript.split(" "))>7 and sess.transcript.split(" ")[-1] in END_WORDS:
                            #     print("[끊고 전송하기] - ", sess.transcript)
                            #     await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript.strip(), "is_final": True}))
                            #     sess.audios = np.empty(0, dtype=np.float32)
                            #     sess.translate_q.put_nowait(sess.transcript)
                            #     sess.transcript = ""
                            #     continue

                            if vad_event == "end" and sess.transcript != "":
                                # 이때까지 자동으로 script 따던게 있을테니 그걸 리턴한다.
                                print("[Voice End] - ", sess.transcript)
                                await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript.strip(), "is_final": True}))
                                sess.current_audio_state = "none"
                                sess.audios = np.empty(0, dtype=np.float32)
                                sess.end_scripting_time = time.time()%1000
                                
                                try:
                                    sess.translate_q.put_nowait(sess.transcript)
                                except:
                                    print("\n\n\nMax translation queue!\n\n")
                                # arm_end_timer(sess, delay=3)
                                sess.transcript = ""
                                continue
                            
                            if sess.buf_count%6==5 and sess.current_audio_state == "start":
                                st = time.time()
                                sess.audios = sess.audios[-WHISPER_SR*20:]
                                pcm_bytes = (np.clip(sess.audios, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

                                # --- NEW: 비동기 큐로 넘겨서 메인 루프가 안 멈춤 ---
                                try:
                                    sess.stt_in_q.put_nowait(pcm_bytes)
                                except asyncio.QueueFull:
                                    # 오래된 것 하나 버리고 최신으로 치환 (지연 누적 방지)
                                    try:
                                        _ = sess.stt_in_q.get_nowait()
                                        sess.stt_in_q.task_done()
                                    except asyncio.QueueEmpty:
                                        pass
                                    try:
                                        sess.stt_in_q.put_nowait(pcm_bytes)
                                    except asyncio.QueueFull:
                                        pass
                                sess.buf_count = 0
                                sess.buf_length = 0

                    except Exception as e:
                     dprint("Error : ", e)
                
                # 3) 커밋 신호 전달 (chunk 경계) = 현재 세팅에서는 VAD를 여기서 검사하기 때문에, 들어올일이 없다.
                elif t == "input_audio_buffer.commit":
                    lprint("input_audio_buffer.commit - ", sess.transcript)
                    if sess.transcript is not None and sess.transcript != "":
                        await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript, "is_final": True}))
                    
                    if sess.transcript is not None and sess.transcript != "":
                        try:
                            sess.translate_q.put_nowait(sess.transcript)
                        except:
                            print("\n\n\nMax translation queue!\n\n")
                        sess.transcript = ""
                        # arm_end_timer(sess, delay=3)
                    
                    sess.current_audio_state = "none"
                    sess.audios = np.empty(0, dtype=np.float32)

                elif t == "test":
                    ct = data.get("current_time")
                    if ct:
                        lprint("network latency : ", time.time()*1000 - ct)
                elif t == "session.close":
                    await ws.send_text(jdumps({
                        "type": "session.close",
                        "payload": {"status": "closed successfully"},
                        "connected_time": time.time() - sess.connection_start_time,
                        "llm_cached_token_count": sess.llm_cached_token_count,
                        "llm_input_token_count": sess.llm_input_token_count,
                        "llm_output_token_count": sess.llm_output_token_count,
                    }))
                    break

                else:
                    # 필요시 기타 타입 처리
                    pass

            elif msg.get("bytes") is not None:
                # 바이너리로도 보낼 수 있다면 여기서 OAI로 전달하는 변형 가능
                buf: bytes = msg["bytes"]
                await ws.send_text(jdumps({
                    "type": "binary_ack",
                    "payload": {"received_bytes": len(buf)}
                }))

    except WebSocketDisconnect:
        pass
    finally:
        await teardown_session(sess)
        sessions.pop(id(ws), None)

async def stt_out_consumer(sess: Session):
    while sess.running:
        msg = await sess.stt_out_q.get()  # {"type":"delta","text":...}
        try:
            newText = (msg or {}).get("text", "") or ""
            # 기존 필터 로직 유지
            if (len(newText.split(" ")) > 6 and len(set(newText.split(" "))) < 2) or newText in ["감사합니다.", "시청해주셔서 감사합니다."]:
                sess.audios = np.empty(0, dtype=np.float32)
                sess.buf_count = 0
                sess.buf_length = 0
                continue
            if sess.current_audio_state != 'none':
                sess.transcript = text_pr(sess.transcript, newText)
                await sess.out_q.put(jdumps({"type": "delta", "text": sess.transcript, "is_final": False}))
        finally:
            sess.stt_out_q.task_done()

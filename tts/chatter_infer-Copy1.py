import torch
import asyncio
import threading
import inspect
from app.session import Session
from llm.grok import chorok_nudge
from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
from librosa.util import normalize
from utils.process import pcm16_b64
from concurrent.futures import ThreadPoolExecutor
import orjson as json
import time
import torchaudio
import re
import random  # ✅ 추가

ENC_EXEC = ThreadPoolExecutor(max_workers=6)
DEFAULT_VOICE_PATH = "./samples/elevenlabs4.mp3"
DEFAULT_KOREAN_VOICE_PATH = "./samples/shogun.wav"

tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

def jdumps(o): return json.dumps(o).decode()

# ✅ 문두 스타터 목록 추가
COMMON_STARTERS = [
    "Yeah.. ",
    "Yep.. ",
    "Nah.. ",
    "Right.. ",
    "Okay.. ",
    "Alright.. ",
    "Well.. ",
    "So, ",
    "Anyway, ",
    "By the way, ",
    "Actually, ",
    "Honestly, ",
    "Seriously, ",
    "Basically, ",
    "Like",
    "You know, ",
    "I mean, ",
    "I guess, ",
    "I think, ",
    "Apparently, ",
    "Obviously, ",
    "Literally, ",
    "Maybe, ",
    "Probably, ",
    "Exactly, ",
    "Sure, ",
    "Uh...",
    "Uhm...",
    "Ah...",
    "Oh!"
]

@torch.inference_mode()
async def chatter_streamer(sess: Session):
    print("[chatter_streamer] START")
    try:
        loop = asyncio.get_running_loop()
        sr = 24000
        OVERLAP = int(0.03 * sr)

        async def emit_chunk_b64(wav_chunk: torch.Tensor, is_final: bool = False):
            if wav_chunk is None or wav_chunk.numel() == 0:
                sess.out_q.put_nowait(jdumps({
                    "type": "tts_audio",
                    "format": "pcm16le",
                    "sample_rate": sr,
                    "channels": 1,
                    "audio": "",
                    "isFinal": is_final,
                }))
                return
            b64 = await loop.run_in_executor(ENC_EXEC, pcm16_b64, wav_chunk)
            sess.out_q.put_nowait(jdumps({
                "type": "tts_audio",
                "format": "pcm16le",
                "sample_rate": sr,
                "channels": 1,
                "audio": b64,
                "isFinal": is_final,
                "server_ts": int(time.time() * 1000),
            }))

        def start_tts_producer_in_thread(text_chunk: str, ref_audio, out_q: asyncio.Queue):
            stop_evt = sess.tts_stop_event
            text_chunk = re.sub(r"\\n", " ... ", text_chunk)
            text_chunk = re.sub(r"\n", " ... ", text_chunk)
            print(text_chunk, "\n")

            async def produce():
                try:
                    agen = tts_model.generate_stream(
                        text_chunk,
                        audio_prompt_path=ref_audio,
                        language_id=sess.language,
                        chunk_size=40,
                        exaggeration=0.4,
                        cfg_weight=0.55,
                        temperature=0.75,
                        repetition_penalty=1.3,
                        min_p=0.02,
                        top_p=0.9
                    )
                    if inspect.isasyncgen(agen):
                        async for evt in agen:
                            if stop_evt.is_set():
                                break
                            audio = evt.get("audio")
                            loop.call_soon_threadsafe(out_q.put_nowait, ("chunk", audio))
                    else:
                        for evt in agen:
                            if stop_evt.is_set():
                                break
                            audio = evt.get("audio")
                            loop.call_soon_threadsafe(out_q.put_nowait, ("chunk", audio))

                    loop.call_soon_threadsafe(out_q.put_nowait, ("eos", None))
                except Exception as e:
                    loop.call_soon_threadsafe(out_q.put_nowait, ("error", str(e)))

            def thread_target():
                asyncio.run(produce())

            t = threading.Thread(target=thread_target, daemon=True)
            t.start()

        async def consume_loop():
            while sess.running:
                text_chunk = await sess.tts_in_q.get()
                if not text_chunk or sess.tts_stop_event.is_set():
                    print("[chatter_streamer] TTS stop event is set", text_chunk)
                    continue
                
                # ✅ (추가) 무음 컨트롤 메시지 처리
                if isinstance(text_chunk, tuple) and len(text_chunk) == 2 and text_chunk[0] == "__silence__":
                    try:
                        silence_sec = float(text_chunk[1])*0.4
                        if silence_sec > 0:
                            num_samples = int(silence_sec * sr)
                            silence_wav = torch.zeros(1, num_samples, dtype=torch.float32)
                            
                            await emit_chunk_b64(silence_wav, is_final=False)
                            print(f"[silence] sent {silence_sec:.2f}s")
                    except Exception as e:
                        print(f"[silence] error: {e}")
                    continue

                # === 참조 오디오 준비 ===
                if hasattr(sess, "ref_audios") and not getattr(sess, "ref_audios").empty():
                    ref_audio = sess.ref_audios.get()
                    sess.ref_audios.put(ref_audio)
                    ref_audio = normalize(ref_audio[-int(16000 * 15):])
                else:
                    ref_audio = DEFAULT_VOICE_PATH if sess.language != 'ko' else DEFAULT_KOREAN_VOICE_PATH

                cancel_silence_nudge(sess)
                
                # === 스레드 → 메인 루프 청크 큐 ===
                tts_chunk_q: asyncio.Queue = asyncio.Queue(maxsize=6)

                # === (추가) 문두 스타터 감지 → 샘플 wav 선송출 ===
                stripped = text_chunk.lstrip()
                matched = None
                for s in COMMON_STARTERS:
                    if stripped.startswith(s):
                        matched = s
                        break

                if matched is not None:
                    token = matched.lower().strip().replace(".", "").replace(",", "")
                    idx = random.choice([0, 1, 2])
                    sample_path = f"./samples/{token}_{idx}.wav"
                    try:
                        wav, sr_file = torchaudio.load(sample_path)  # (ch, T)
                        if wav.dim() == 2 and wav.size(0) > 1:
                            wav = wav.mean(dim=0, keepdim=True)  # mono
                        if sr_file != sr:
                            wav = torchaudio.functional.resample(wav, sr_file, sr)
                        asyncio.create_task(emit_chunk_b64(wav, is_final=False))
                    except Exception as e:
                        print(f"[starter] failed to load '{sample_path}': {e}")

                # === 스트리밍 상태 ===
                last_length = 0
                last_tail: torch.Tensor | None = None

                start_time = time.time()

                # === TTS 생산자(스레드) 시작 (샘플 송출과 동시 진행) ===
                if matched is not None:
                    text_chunk = re.sub(matched, '', text_chunk[:10]) + text_chunk[10:]
                
                start_tts_producer_in_thread(text_chunk, ref_audio, tts_chunk_q)
                start_sending_at = 0
                total_audio_seconds = 0

                # === 소비: 큐에서 꺼내서 교차페이드 & 전송 ===
                while True:
                    if sess.tts_stop_event.is_set():
                        try:
                            while True:
                                _ = tts_chunk_q.get_nowait()
                                tts_chunk_q.task_done()
                        except asyncio.QueueEmpty:
                            pass
                        break
                    evt_type, payload = await tts_chunk_q.get()

                    if evt_type == "chunk":
                        wav: torch.Tensor = payload  # (ch, T)
                        if wav is None or wav.numel() == 0:
                            await asyncio.sleep(0)
                            continue
                        new_total = wav.shape[-1]
                        delta = new_total - last_length
                        if delta <= 0:
                            await asyncio.sleep(0)
                            continue
                        new_part = wav[:, last_length:new_total]

                        # --- 교차페이드 ---
                        if last_tail is None:
                            out_chunk = new_part
                        else:
                            L = min(OVERLAP, new_part.shape[-1], last_tail.shape[-1])
                            if L > 0:
                                dtype = wav.dtype
                                device = wav.device
                                fade_in  = torch.linspace(0, 1, L, device=device, dtype=dtype)
                                fade_out = 1.0 - fade_in
                                mixed = last_tail[:, -L:] * fade_out + new_part[:, :L] * fade_in
                                tail  = new_part[:, L:]
                                out_chunk = torch.cat([mixed, tail], dim=-1)
                            else:
                                out_chunk = new_part

                        new_tail_start = max(0, new_total - OVERLAP)
                        last_tail = wav[:, new_tail_start:new_total].detach()

                        print(f"[TTS {(new_total-last_length)/24000:.3f}] - takes {time.time() - start_time:.3f}")
                        total_audio_seconds += (new_total-last_length)/24000
                        await emit_chunk_b64(out_chunk, is_final=False)

                        if start_sending_at == 0:
                            start_sending_at = time.time()
                        
                        last_length = new_total
                        await asyncio.sleep(0)

                    elif evt_type == "eos":
                        if last_tail is not None and last_tail.numel() > 0:
                            await emit_chunk_b64(last_tail, is_final=True)
                        else:
                            await emit_chunk_b64(None, is_final=True)

                        taken = time.time() - start_sending_at
                        remaining_until_audio_end = total_audio_seconds - taken + 1
                        print(f"Taken : {taken:.3f}, Total audio : {total_audio_seconds:.3f}, Remain : {remaining_until_audio_end:.3f}")
                        
                        schedule_silence_nudge(sess, delay=3.0, remain=remaining_until_audio_end)
                        
                        break

                    elif evt_type == "error":
                        print("🥊 [chatter_streamer] TTS error:", payload)
                        sess.out_q.put_nowait(jdumps({"type": "tts_error", "message": payload}))
                        break

        await consume_loop()

    except asyncio.CancelledError:
        print("[chatter_streamer] CANCELLED")
        raise
    except Exception as e:
        print("[chatter_streamer] ERROR:", e)
        await sess.out_q.put(jdumps({"type": "tts_error", "message": str(e)}))
    finally:
        print("[chatter_streamer] END")

async def proactive_say(sess: Session):
    """유저가 5초간 말이 없을 때 먼저 한마디 하는 함수 (chorok_nudge 사용)"""
    loop = asyncio.get_running_loop()

    def run_blocking():
        return chorok_nudge(
            prev_scripts=sess.transcripts[-6:],
            prev_answers=sess.outputs[-6:],
            language=sess.language,
            name=sess.name,
            current_time=sess.current_time,
        )

    try:
        output = await loop.run_in_executor(None, run_blocking)
        tuples = (output.get("text", "") or "")
        print(f"[Nudge Answers] - {tuples!r}")
        if tuples[0] is None or tuples[0] == '':
            return
        if tuples[1] == 'wait':
            return
        text = tuples[0]
        
        sess.answer = text.strip()
        sess.outputs[-1] = sess.outputs[-1] + " (User silence for five seconds) " + text

    except Exception as e:
        print("NUDGE ERROR : ", e)
    
    loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, text)
    loop.call_soon_threadsafe(
        sess.out_q.put_nowait,
        jdumps({"type": "translated", "script": "", "text": text, "is_final": True})
    )

def cancel_silence_nudge(sess: Session):
    """예약된 넛지 타이머가 있으면 취소"""
    task = getattr(sess, "silence_nudge_task", None)
    if task and not task.done():
        task.cancel()
    setattr(sess, "silence_nudge_task", None)

def schedule_silence_nudge(sess: Session, delay: float = 5.0, remain: float = 1.0):
    cancel_silence_nudge(sess)

    async def waiter():
        try:
            await asyncio.sleep(remain)
            print("말이 끝났다고 판단\n")
            if getattr(sess, "current_audio_state", "none") != "none":
                print("말 리턴")
                return
            await asyncio.sleep(delay)
            print("이제 선톡 날리기\n")
            if getattr(sess, "current_audio_state", "none") == "none":
                await proactive_say(sess)
        except asyncio.CancelledError:
            pass

    # 새 타이머 등록
    sess.silence_nudge_task = asyncio.create_task(waiter())

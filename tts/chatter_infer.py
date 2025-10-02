import torch
import asyncio
import threading
import inspect
from app.session import Session
from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
from librosa.util import normalize
from utils.process import pcm16_b64
from concurrent.futures import ThreadPoolExecutor
import orjson as json
import time
import torchaudio
import re

ENC_EXEC = ThreadPoolExecutor(max_workers=6)
DEFAULT_VOICE_PATH = "./utils/output_full.wav"
DEFAULT_KOREAN_VOICE_PATH = "./utils/shogun.wav"

tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

def jdumps(o): return json.dumps(o).decode()

@torch.inference_mode()
async def chatter_streamer(sess: Session):
    print("[chatter_streamer] START")
    try:
        loop = asyncio.get_running_loop()
        sr = 24000
        OVERLAP = int(0.03 * sr)

        async def emit_chunk_b64(wav_chunk: torch.Tensor, is_final: bool = False):
            """PCM16 base64 인코딩을 별도 풀에서 처리해 이벤트루프를 막지 않음"""
            if wav_chunk is None or wav_chunk.numel() == 0:
                # 빈 바디로 종료 신호만 보낼 수도 있음
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
                        chunk_size=45,
                        exaggeration=0.4,
                        cfg_weight=0.5,
                        temperature=0.7,
                        repetition_penalty=1.3,
                        min_p=0.02,
                        top_p=0.9
                    )
                    # async / sync 모두 방어
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
                # 스레드 전용 이벤트 루프 생성/실행
                asyncio.run(produce())

            t = threading.Thread(target=thread_target, daemon=True)
            t.start()

        async def consume_loop():
            while sess.running:
                text_chunk = await sess.tts_in_q.get()
                if not text_chunk or not text_chunk.strip() or sess.tts_stop_event.is_set():
                    print("[chatter_streamer] TTS stop event is set", text_chunk)
                    continue

                # === 참조 오디오 준비 ===
                if hasattr(sess, "ref_audios") and not getattr(sess, "ref_audios").empty():
                    ref_audio = sess.ref_audios.get()
                    sess.ref_audios.put(ref_audio)
                    # 최근 15초만, 정규화
                    ref_audio = normalize(ref_audio[-int(16000 * 15):])
                else:
                    ref_audio = DEFAULT_VOICE_PATH if sess.language != 'ko' else DEFAULT_KOREAN_VOICE_PATH

                # === 스레드 → 메인 루프 청크 큐 ===
                tts_chunk_q: asyncio.Queue = asyncio.Queue(maxsize=6)

                # === 스트리밍 상태 ===
                last_length = 0
                last_tail: torch.Tensor | None = None

                start_time = time.time()

                # === TTS 생산자(스레드) 시작 ===
                start_tts_producer_in_thread(text_chunk, ref_audio, tts_chunk_q)

                # === 소비: 큐에서 꺼내서 교차페이드 & 전송 ===
                while True:
                    # 🔴 인터럽트 즉시 중단
                    if sess.tts_stop_event.is_set():
                        # 큐 비우기
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

                        new_part = wav[:, last_length:new_total]  # 새로 추가된 영역
                        
                        print(f"[TTS {(new_total-last_length)/24000:.3f}] - takes {time.time() - start_time:.3f}")
                        # start_time = time.time()

                        # --- 교차페이드(OVERLAP) ---
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

                        # 다음 교차페이드를 위한 tail 갱신
                        new_tail_start = max(0, new_total - OVERLAP)
                        last_tail = wav[:, new_tail_start:new_total].detach()

                        # 전송 (인코딩은 스레드풀)
                        await emit_chunk_b64(out_chunk, is_final=False)

                        last_length = new_total
                        # 다른 코루틴(WS/STT)에 양보
                        await asyncio.sleep(0)

                    elif evt_type == "eos":
                        # 남은 tail flush + 종료 패킷
                        if last_tail is not None and last_tail.numel() > 0:
                            await emit_chunk_b64(last_tail, is_final=True)
                        else:
                            await emit_chunk_b64(None, is_final=True)
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

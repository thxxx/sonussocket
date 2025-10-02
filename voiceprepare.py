from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
import time
from tqdm import tqdm
from IPython.display import Audio
import torchaudio
import re

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

common_starters = [
    "Yeah",
    "Yep",
    "Nah",
    "Right",
    "Okay",
    "Alright",
    "Well",
    "So",
    "Anyway",
    "By the way",
    "Actually",
    "Honestly",
    "Seriously",
    "Basically",
    "Like",
    "You know",
    "I mean",
    "I guess",
    "I think",
    "Apparently",
    "Obviously",
    "Literally",
    "Maybe",
    "Probably",
    "Exactly",
    "Sure",
    "Uh...",
    "Uhm...",
    "Ah...",
    "Oh!"
]

total_seconds = 0
total_latency = 0
for text in common_starters:
    st = time.time()

    wav = model.generate(
        text, 
        audio_prompt_path='./samples/output.wav', 
        language_id='en',
        exaggeration=0.4,
        cfg_weight=0.6,
        temperature=0.7,
        repetition_penalty=1.3,
        min_p=0.02,
        top_p=0.9
    )
    total_seconds += wav.shape[-1]/24000
    total_latency += time.time() - st
    # print(text)
    # print(f"[For {wav.shape[-1]/24000}s] taken: {time.time() - st:.3f}")
    display(Audio(wav, rate=24000))
    torchaudio.save(f"{re.sub('\.\.\.|!', '', text)}.wav", waveform, sample_rate=sr)

print(f"1초 생성에 {total_latency/total_seconds:.3f}초 걸림")
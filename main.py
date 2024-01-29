from typing import Literal, Generator
import asyncio
import struct

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import soundfile as sf
import numpy as np
from pydub import AudioSegment



class AudioRequest(BaseModel):

    output_format: Literal["mp3", "wav"] = "mp3"



def add_wav_header(
    sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16
) -> bytes:
    len_data = 2**32 - 1 - 36
    # WAV 文件头的相关信息
    chunk_id = b"RIFF"
    chunk_size = len_data + 36
    format_ = b"WAVE"
    subchunk1_id = b"fmt "
    subchunk1_size = 16
    audio_format = 1  # PCM 格式
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    subchunk2_id = b"data"
    subchunk2_size = len_data
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        chunk_id, chunk_size, format_,
        subchunk1_id, subchunk1_size, audio_format, channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        subchunk2_id, subchunk2_size
    )
    return header


def postprocess(wav: np.ndarray, sample_width: int, channels: int) -> np.ndarray:
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    wav = wav.squeeze()
    remind = len(wav) % (sample_width * channels)
    wav = wav[:len(wav) - remind]
    return wav



def infer(
    output_format: str, 
    sample_rate: int, 
    sample_width: int, 
    channels: int,
) -> Generator:
    chunk_size = 10240
    for i in range(0, len(wav), chunk_size):
        chunk = wav[i: i+chunk_size]
        if output_format == "mp3":
            audio = AudioSegment(
                chunk,
                sample_width=sample_width,
                frame_rate=sample_rate,
                channels=channels
            )
            yield audio.export(format=output_format).read()
        elif output_format == "wav":
            if i == 0:
                yield add_wav_header(sample_rate)
            yield chunk.tobytes()


async def generate_audio(
    output_format: str, 
    sample_rate: int,
    sample_width: int, 
    channels: int,
) -> Generator:
    for chunk in infer(output_format, sample_rate, sample_width, channels):
        print("current chunk: ", chunk[:50])
        yield chunk
        await asyncio.sleep(0)



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sample_width = 2
channels = 1
wav, sample_rate = sf.read("sample.wav")
print("raw wav read from file: ", wav, sample_rate)
wav = postprocess(wav, sample_width, channels)
print("wav after post process: ", wav, len(wav))


@app.post("/audio_stream")
async def get_audio(req: AudioRequest):
    if req.output_format == "wav":
        mie = "audio/wav"
    else:
        mie = "audio/mpeg"
    gen = generate_audio(req.output_format, sample_rate, sample_width, channels)
    return StreamingResponse(gen, media_type=mie)

"""
AI helper modules. This package holds reusable AI-related utilities and providers.

The transcribe module is adapted from the original youtube-rag/rag.py example and
contains harmless, well-documented functions to download audio and transcribe using Whisper.

This module should be invoked from a worker process or called by an API endpoint.
"""

import os
import tempfile
from typing import Optional

try:
    from pytube import YouTube
except Exception:
    YouTube = None  # optional dependency for scaffold

try:
    import whisper
except Exception:
    whisper = None


def download_audio_from_youtube(url: str, output_dir: Optional[str] = None) -> str:
    """Download the highest-quality audio stream for a YouTube video.

    Returns the path to the downloaded file.
    This function requires pytube to be installed.
    """
    if YouTube is None:
        raise RuntimeError("pytube is not installed. Install dependencies in api/requirements.txt or ai/requirements.txt")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="tcs_yt_")

    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    if audio_stream is None:
        raise RuntimeError("No audio stream found for this video")

    out_file = audio_stream.download(output_path=output_dir)
    return out_file


def transcribe_file_whisper(path: str, model_name: str = "base") -> str:
    """Transcribe an audio file using whisper and return the transcript text.

    This function requires the whisper package (https://github.com/openai/whisper).
    """
    if whisper is None:
        raise RuntimeError("whisper is not installed. Install via git+https://github.com/openai/whisper.git")

    model = whisper.load_model(model_name)
    result = model.transcribe(path, fp16=False)
    text = result.get("text", "").strip()
    return text


def youtube_to_transcript(url: str, transcript_path: str = "transcripts.txt", model_name: str = "base") -> str:
    """High-level helper: download audio then transcribe and write to path.

    Returns the transcript path.

    This is a synchronous helper for simple worker processes. In production run
    this in a separate worker with timeouts and retries.
    """
    audio_file = download_audio_from_youtube(url)
    transcript = transcribe_file_whisper(audio_file, model_name=model_name)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return transcript_path


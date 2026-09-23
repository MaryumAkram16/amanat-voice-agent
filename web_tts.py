import asyncio

import edge_tts

DEFAULT_VOICE = "ur-PK-UzmaNeural"  # female Urdu (Pakistan). Alternative: ur-PK-AsadNeural (male)


async def _synthesize_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def synthesize(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Text -> mp3 bytes. Safe to call from a background thread (used by session.py)."""
    return asyncio.run(_synthesize_async(text, voice))

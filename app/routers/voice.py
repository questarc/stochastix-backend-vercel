from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ..deps import get_audio_store, get_voice_agent
from ..schemas import AudioInfo, VoiceRequest
from ..voice import AudioStore, VoiceGenAgent

router = APIRouter(tags=["voice"])


@router.post("/voice", response_model=AudioInfo)
async def create_audio(req: VoiceRequest, agent: VoiceGenAgent = Depends(get_voice_agent)):
    clip = await run_in_threadpool(agent.create, req.text, req.account_id)
    return clip.to_dict()


@router.get("/voice/{audio_id}")
def download_audio(audio_id: str, download: bool = False, store: AudioStore = Depends(get_audio_store)):
    clip = store.get(audio_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="This audio file has expired. Create it again.")
    disposition = "attachment" if download else "inline"
    headers = {"Content-Disposition": f'{disposition}; filename="{clip.file_name}"'}
    return Response(content=clip.data, media_type="audio/wav", headers=headers)

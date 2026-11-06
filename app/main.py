"""FastAPI 服务：上传 wav → 返回匿名音频 + 转录文本 + F0 摘要。"""
import os
import time
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import AnonymizationPipeline

app = FastAPI(title="Voice Anonymization", version="0.1")
pipe = AnonymizationPipeline(out_dir="outputs")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/anonymize")
async def anonymize(file: UploadFile = File(...)):
    # 保存上传
    tmp_in = os.path.join("outputs", f"in_{int(time.time())}_{file.filename}")
    os.makedirs("outputs", exist_ok=True)
    with open(tmp_in, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out_name = f"anon_{int(time.time())}.wav"
    try:
        result = pipe.run(tmp_in, out_name=out_name)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)

    result["download_url"] = f"/outputs/{out_name}"
    result["original_url"] = f"/static/{os.path.basename(tmp_in)}" if False else None
    return result


# 挂载 outputs 以便下载合成音频
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

"""FastAPI 服务：上传 wav → 返回匿名音频 + 转录文本 + F0 摘要。"""
import os
import time
import json
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import AnonymizationPipeline

app = FastAPI(title="Voice Anonymization", version="0.2")
pipe = AnonymizationPipeline(out_dir="outputs")

# 允许跨域（前端平台在别的域名下调本接口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 公网基地址（华为云部署时用；本地开发时自动用 localhost:8000）
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "http://localhost:8000")

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


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """平台标准接口：POST multipart/form-data → 返回 artifacts JSON。"""
    tmp_in = os.path.join("outputs", f"in_{int(time.time())}_{file.filename}")
    os.makedirs("outputs", exist_ok=True)
    with open(tmp_in, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out_name = f"anon_{int(time.time())}.wav"
    try:
        result = pipe.run(tmp_in, out_name=out_name)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)

    # 写一份 JSON 报告
    report_name = f"report_{int(time.time())}.json"
    report_path = os.path.join("outputs", report_name)
    report = {
        "transcribed_text": result["text"],
        "f0_mean_hz": round(result["f0_mean"], 2),
        "n_f0_frames": result["n_frames"],
        "emotion_prototype_source": result["prototype_source"],
        "attention_weight_shape": result["attn_shape"],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "message": "语音匿名化处理完成",
        "artifacts": [
            {
                "kind": "audio",
                "label": "匿名后音频（固定伪说话人音色）",
                "filename": out_name,
                "url": f"{PUBLIC_BASE}/outputs/{out_name}",
            },
            {
                "kind": "json",
                "label": "处理报告（转录文本 + F0 + 情感原型）",
                "filename": report_name,
                "url": f"{PUBLIC_BASE}/outputs/{report_name}",
            },
        ],
    }

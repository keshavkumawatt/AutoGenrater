from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image, ImageDraw, ImageEnhance
from openai import OpenAI
import numpy as np
import os
import uuid
import requests
from urllib.parse import quote_plus
from moviepy import VideoClip, AudioFileClip, CompositeVideoClip

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_URL = os.getenv("SERVER_BASE_URL", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


class ContentRequest(BaseModel):
    message: str


class ImageRequest(BaseModel):
    prompt: str


class VideoRequest(BaseModel):
    prompt: str
    duration: int = 10


# ✅ SAFE CLIENT INIT
image_client = None
text_client = None

if HF_TOKEN:
    image_client = InferenceClient(provider="wavespeed", api_key=HF_TOKEN)
    text_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
else:
    print("HF_TOKEN missing")


def get_url(request: Request, file):
    if BASE_URL:
        return f"{BASE_URL}/outputs/{file}"
    return f"{request.base_url}outputs/{file}"


@app.get("/")
def home():
    return {"message": "API Running 🚀"}


# ================= CONTENT =================
@app.post("/generate-content")
def generate_content(req: ContentRequest):
    try:
        if not text_client:
            return {"success": False, "error": "HF_TOKEN missing"}

        res = text_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[
                {"role": "user", "content": req.message}
            ],
        )

        return {
            "success": True,
            "reply": res.choices[0].message.content
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ================= IMAGE =================
@app.post("/generate-image")
def generate_image(req: ImageRequest, request: Request):
    try:
        file = f"{uuid.uuid4()}.png"
        path = os.path.join(OUTPUT_DIR, file)

        if image_client:
            try:
                img = image_client.text_to_image(
                    req.prompt,
                    model="black-forest-labs/FLUX.1-dev"
                )
                img.save(path)

                return {
                    "success": True,
                    "url": get_url(request, file)
                }
            except:
                pass

        # fallback
        encoded = quote_plus(req.prompt)
        r = requests.get(f"https://image.pollinations.ai/prompt/{encoded}")

        with open(path, "wb") as f:
            f.write(r.content)

        return {
            "success": True,
            "url": get_url(request, file)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ================= VIDEO =================
@app.post("/generate-video")
def generate_video(req: VideoRequest, request: Request):
    try:
        uid = str(uuid.uuid4())

        img_path = f"{OUTPUT_DIR}/{uid}.png"
        audio_path = f"{OUTPUT_DIR}/{uid}.mp3"
        video_path = f"{OUTPUT_DIR}/{uid}.mp4"

        # image
        ok = False
        if image_client:
            try:
                img = image_client.text_to_image(req.prompt)
                img.save(img_path)
                ok = True
            except:
                pass

        if not ok:
            encoded = quote_plus(req.prompt)
            r = requests.get(f"https://image.pollinations.ai/prompt/{encoded}")
            with open(img_path, "wb") as f:
                f.write(r.content)

        # audio
        tts = gTTS(req.prompt)
        tts.save(audio_path)

        audio = AudioFileClip(audio_path)

        def make_frame(t):
            img = Image.open(img_path).resize((1280, 720))
            return np.array(img)

        clip = VideoClip(make_frame, duration=req.duration)
        clip = clip.set_audio(audio)

        clip.write_videofile(video_path, fps=24)

        return {
            "success": True,
            "url": get_url(request, os.path.basename(video_path))
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

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
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL")

app = FastAPI(title="AI Chatbot + Image + Video Generator")

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


image_client = None
text_client = None

if HF_TOKEN:
    image_client = InferenceClient(provider="wavespeed", api_key=HF_TOKEN)

    text_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
else:
    print("WARNING: HF_TOKEN is missing. Add HF_TOKEN in Render Environment Variables.")


def get_base_url(request: Request):
    if SERVER_BASE_URL:
        return SERVER_BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/")


def _fetch_ai_image(prompt: str, save_path: str) -> bool:
    if image_client:
        try:
            image = image_client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev"
            )
            image.save(save_path)
            return True
        except Exception as e:
            print("HuggingFace image failed:", str(e))

    try:
        encoded = quote_plus(prompt)
        response = requests.get(
            f"https://image.pollinations.ai/prompt/{encoded}",
            timeout=60
        )

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True

    except Exception as e:
        print("Pollinations image failed:", str(e))

    return False


def _make_animated_clip(image_path: str, duration: float) -> VideoClip:
    W, H = 1280, 720
    FPS = 24

    pil_img = Image.open(image_path).convert("RGB").resize((W, H), Image.LANCZOS)

    CANVAS_W, CANVAS_H = int(W * 1.25), int(H * 1.25)
    pil_large = pil_img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
    large_arr = np.array(pil_large)

    vignette = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(vignette)

    for i in range(min(W, H) // 2):
        alpha = int(255 * (i / (min(W, H) / 2)) ** 0.7)
        draw.ellipse([i, i, W - i, H - i], outline=alpha)

    vignette_arr = np.array(vignette) / 255.0

    def make_frame(t):
        progress = min(t / duration, 1)

        scale = 1.0 + 0.15 * progress

        crop_w = int(W / scale)
        crop_h = int(H / scale)

        x_offset = int((CANVAS_W - crop_w) * (0.35 + 0.12 * progress))
        y_offset = int((CANVAS_H - crop_h) * (0.45 + 0.04 * np.sin(progress * np.pi)))

        x_offset = max(0, min(x_offset, CANVAS_W - crop_w))
        y_offset = max(0, min(y_offset, CANVAS_H - crop_h))

        cropped = large_arr[
            y_offset:y_offset + crop_h,
            x_offset:x_offset + crop_w
        ]

        frame_pil = Image.fromarray(cropped).resize((W, H), Image.LANCZOS)

        brightness = 1.0 + 0.05 * np.sin(2 * np.pi * t / max(duration, 1))
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)

        frame = np.array(frame_pil).astype(np.float64)

        for c in range(3):
            frame[:, :, c] = frame[:, :, c] * (0.65 + 0.35 * vignette_arr)

        if t < 1.2:
            fade = t / 1.2
            frame = frame * fade

        return frame.clip(0, 255).astype(np.uint8)

    clip = VideoClip(make_frame, duration=duration)
    clip = clip.with_fps(FPS)

    return clip


@app.get("/")
def home(request: Request):
    base_url = get_base_url(request)

    return {
        "success": True,
        "message": "AI Chatbot + Image + Video API Running",
        "docs": f"{base_url}/docs"
    }


@app.post("/generate-content")
def generate_content(request_data: ContentRequest):
    try:
        if not text_client:
            return {
                "success": False,
                "error": "HF_TOKEN is missing. Add HF_TOKEN in Render Environment Variables."
            }

        user_message = request_data.message.strip()

        if not user_message:
            return {"success": False, "error": "Message is required"}

        completion = text_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a smart AI chatbot and content generator.

Rules:
- Always reply in English
- Give dynamic responses based on user input
- If user asks for a story, give a story
- If user asks for caption, give caption
- If user asks for script, give script
- If user chats normally, reply like a chatbot
- Keep answers clear, useful, and engaging
"""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
        )

        return {
            "success": True,
            "message": user_message,
            "reply": completion.choices[0].message.content
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/generate-image")
def generate_image(request_data: ImageRequest, request: Request):
    try:
        base_url = get_base_url(request)

        prompt = request_data.prompt.strip()

        if not prompt:
            return {"success": False, "error": "Prompt is required"}

        file_name = f"{uuid.uuid4()}.png"
        path = os.path.join(OUTPUT_DIR, file_name)

        if image_client:
            try:
                image = image_client.text_to_image(
                    prompt,
                    model="black-forest-labs/FLUX.1-dev"
                )
                image.save(path)

                return {
                    "success": True,
                    "source": "huggingface",
                    "file": file_name,
                    "url": f"{base_url}/outputs/{file_name}"
                }

            except Exception as e:
                print("HuggingFace image failed:", str(e))

        encoded = quote_plus(prompt)
        response = requests.get(
            f"https://image.pollinations.ai/prompt/{encoded}",
            timeout=60
        )

        if response.status_code != 200:
            return {
                "success": False,
                "error": "Image generation failed"
            }

        with open(path, "wb") as f:
            f.write(response.content)

        return {
            "success": True,
            "source": "pollinations",
            "file": file_name,
            "url": f"{base_url}/outputs/{file_name}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/generate-video")
def generate_video(request_data: VideoRequest, request: Request):
    try:
        base_url = get_base_url(request)

        prompt = request_data.prompt.strip()

        if not prompt:
            return {"success": False, "error": "Prompt is required"}

        uid = uuid.uuid4()

        image_path = os.path.join(OUTPUT_DIR, f"{uid}_image.png")
        audio_path = os.path.join(OUTPUT_DIR, f"{uid}_audio.mp3")
        video_path = os.path.join(OUTPUT_DIR, f"{uid}_video.mp4")

        image_ok = _fetch_ai_image(prompt, image_path)

        if not image_ok:
            return {
                "success": False,
                "error": "Image generation failed. Video cannot be created."
            }

        tts = gTTS(text=prompt, lang="en")
        tts.save(audio_path)

        audio_clip = AudioFileClip(audio_path)

        video_duration = request_data.duration

        if video_duration < 3:
            video_duration = 3

        animated_clip = _make_animated_clip(image_path, video_duration)

        final_clip = CompositeVideoClip([animated_clip])
        final_clip = final_clip.with_audio(audio_clip)

        final_clip.write_videofile(
            video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        audio_clip.close()
        animated_clip.close()
        final_clip.close()

        video_name = os.path.basename(video_path)
        image_name = os.path.basename(image_path)

        return {
            "success": True,
            "file": video_name,
            "url": f"{base_url}/outputs/{video_name}",
            "preview_image_url": f"{base_url}/outputs/{image_name}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
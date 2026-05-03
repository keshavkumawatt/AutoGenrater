from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image
from openai import OpenAI
import numpy as np
import os
import uuid
import requests
from urllib.parse import quote_plus
from moviepy import VideoClip, AudioFileClip, CompositeVideoClip

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "")

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


def get_file_url(request: Request, filename: str):
    if SERVER_BASE_URL:
        return f"{SERVER_BASE_URL.rstrip('/')}/outputs/{filename}"
    return f"{str(request.base_url).rstrip('/')}/outputs/{filename}"


def fallback_content(message: str):
    return f"""
Here is a simple content idea based on your message: "{message}"

Title: A Student's Journey

Once there was a student who wanted to become successful but faced many challenges. Every day, the student worked hard, learned from mistakes, and kept moving forward. Slowly, confidence started growing. The student understood that success does not come in one day, but daily effort can change everything.

Moral: Hard work, patience, and consistency can turn dreams into reality.
"""


@app.get("/")
def home(request: Request):
    return {
        "success": True,
        "message": "AI Chatbot + Image + Video API Running",
        "docs": f"{str(request.base_url).rstrip('/')}/docs"
    }


@app.post("/generate-content")
def generate_content(req: ContentRequest):
    try:
        user_message = req.message.strip()

        if not user_message:
            return {"success": False, "error": "Message is required"}

        if not text_client:
            return {
                "success": True,
                "source": "fallback",
                "message": user_message,
                "reply": fallback_content(user_message)
            }

        completion = text_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart AI chatbot and content generator. Always reply in English."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
        )

        return {
            "success": True,
            "source": "huggingface",
            "message": user_message,
            "reply": completion.choices[0].message.content
        }

    except Exception as e:
        return {
            "success": True,
            "source": "fallback",
            "message": req.message,
            "reply": fallback_content(req.message),
            "warning": str(e)
        }


@app.post("/generate-image")
def generate_image(req: ImageRequest, request: Request):
    try:
        prompt = req.prompt.strip()

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
                    "url": get_file_url(request, file_name)
                }

            except Exception:
                pass

        encoded = quote_plus(prompt)
        response = requests.get(
            f"https://image.pollinations.ai/prompt/{encoded}",
            timeout=60
        )

        if response.status_code != 200:
            return {"success": False, "error": "Image generation failed"}

        with open(path, "wb") as f:
            f.write(response.content)

        return {
            "success": True,
            "source": "pollinations",
            "file": file_name,
            "url": get_file_url(request, file_name)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def fetch_image_for_video(prompt: str, image_path: str):
    if image_client:
        try:
            image = image_client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev"
            )
            image.save(image_path)
            return True
        except Exception:
            pass

    try:
        encoded = quote_plus(prompt)
        response = requests.get(
            f"https://image.pollinations.ai/prompt/{encoded}",
            timeout=60
        )

        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            return True

    except Exception:
        pass

    return False


@app.post("/generate-video")
def generate_video(req: VideoRequest, request: Request):
    audio_clip = None
    video_clip = None
    final_clip = None

    try:
        prompt = req.prompt.strip()

        if not prompt:
            return {"success": False, "error": "Prompt is required"}

        duration = req.duration
        if duration < 3:
            duration = 3
        if duration > 15:
            duration = 15

        uid = str(uuid.uuid4())

        image_path = os.path.join(OUTPUT_DIR, f"{uid}_image.png")
        audio_path = os.path.join(OUTPUT_DIR, f"{uid}_audio.mp3")
        video_path = os.path.join(OUTPUT_DIR, f"{uid}_video.mp4")

        image_ok = fetch_image_for_video(prompt, image_path)

        if not image_ok:
            return {
                "success": False,
                "error": "Image generation failed. Video cannot be created."
            }

        tts = gTTS(text=prompt, lang="en")
        tts.save(audio_path)

        audio_clip = AudioFileClip(audio_path)

        base_img = Image.open(image_path).convert("RGB").resize((1280, 720))
        base_array = np.array(base_img)

        def make_frame(t):
            return base_array

        video_clip = VideoClip(make_frame, duration=duration).with_fps(24)

        # ✅ MoviePy v2 fix: set_audio ❌, with_audio ✅
        final_clip = CompositeVideoClip([video_clip]).with_audio(audio_clip)

        final_clip.write_videofile(
            video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        video_name = os.path.basename(video_path)
        image_name = os.path.basename(image_path)

        return {
            "success": True,
            "file": video_name,
            "url": get_file_url(request, video_name),
            "preview_image_url": get_file_url(request, image_name)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        try:
            if audio_clip:
                audio_clip.close()
            if video_clip:
                video_clip.close()
            if final_clip:
                final_clip.close()
        except Exception:
            pass

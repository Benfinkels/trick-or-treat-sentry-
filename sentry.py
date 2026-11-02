#!/usr/bin/env python3
"""
sentry.py — Real-Time Edge AI Halloween Trick-or-Treater Sentry

Architecture:
1. Low-Power Stage 1 Filter: PIL grayscale 160x120 pixel diffing to filter static frames (<5ms, 0 GPU load).
2. Stage 2 Multimodal Vision: On motion delta > threshold, dispatches full frame to local Gemma 4 Multimodal Vision.
3. Adaptive Optics: Removes purple/magenta IR color casts caused by camera glass reflections and night-mode sensors.
4. Rich Real-Time Alerts: Sends rich Discord embeds with live snapshots, visitor counts, and costume classifications.
5. Zero Cloud / 100% Private: All inference runs locally on consumer GPU hardware with $0 API token costs.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import ssl
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from PIL import Image, ImageChops, ImageFile, ImageStat

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("halloween_sentry")


def load_env(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load key-value pairs from a .env file."""
    env = {}
    candidates = [
        env_path,
        Path.cwd() / ".env",
        Path.home() / ".env",
    ]
    target = next((p for p in candidates if p and p.exists()), None)
    if target:
        for line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    return env


class CameraClient:
    """Client for fetching snapshot frames from an edge camera with HTTP Basic Auth."""

    def __init__(self, url: str, user: str = "", password: str = ""):
        self.url = url
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        handlers: list[Any] = [urllib.request.HTTPSHandler(context=ctx)]
        if user and password:
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, url, user, password)
            handlers.append(urllib.request.HTTPBasicAuthHandler(password_mgr))

        self.opener = urllib.request.build_opener(*handlers)

    def fetch_frame_bytes(self, timeout: float = 4.0) -> bytes:
        """Fetch raw image JPEG bytes from camera endpoint."""
        with self.opener.open(self.url, timeout=timeout) as resp:
            return resp.read()

    @staticmethod
    def downsample_frame(raw_bytes: bytes, resolution: Tuple[int, int] = (160, 120)) -> Image.Image:
        """Downsample image to low-resolution grayscale for instantaneous pixel diffing."""
        img = Image.open(io.BytesIO(raw_bytes)).convert("L")
        return img.resize(resolution)


def compute_frame_difference(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute mean pixel intensity difference between two grayscale frames.
    Executes in <5ms with 0 GPU utilization to reject empty/static frames.
    """
    diff = ImageChops.difference(img1, img2)
    hist = diff.histogram()
    w, h = img1.size
    total_pixels = w * h
    diff_sum = sum(i * count for i, count in enumerate(hist))
    return diff_sum / max(total_pixels, 1)


def fix_color_cast(img_bytes: bytes) -> bytes:
    """
    Remove purple/magenta IR cast through glass or when IR cut-filter is disengaged.
    Near-monochrome night frames are cleanly normalized to grayscale;
    otherwise gray-world white balancing is applied to preserve costume colors.
    """
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        r, g, b = ImageStat.Stat(im).mean
        avg = (r + g + b) / 3.0
        if max(r, g, b) - min(r, g, b) < 2.0:
            return img_bytes  # already neutral

        magenta = (r + b) / 2.0 - g
        if magenta > 6.0 and g < avg:
            im = im.convert("L").convert("RGB")  # IR scene: honest grayscale rendering
        else:
            bands = [
                ch.point(lambda v, k=avg / max(m, 1.0): min(255, int(v * k)))
                for ch, m in zip(im.split(), (r, g, b))
            ]
            im = Image.merge("RGB", bands)

        out = io.BytesIO()
        im.save(out, "JPEG", quality=88)
        return out.getvalue()
    except Exception as e:
        logger.warning(f"Color correction skipped: {e}")
        return img_bytes


def analyze_with_local_vision(
    img_bytes: bytes,
    ollama_url: str = "http://127.0.0.1:11434/api/chat",
    model: str = "gemma4:e2b-it-q4_K_M",
    timeout: float = 20.0,
) -> Dict[str, Any]:
    """
    Dispatch frame to local multimodal model via Ollama with strict JSON schema.
    Uses keep_alive='15m' and think=False for ultra-fast response without PCIe reloading.
    """
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = (
        "Front-door camera frame on Halloween night. Is a person approaching or at the door?\n"
        "Trick-or-treater = costume, mask, candy bucket/bag, or kids.\n"
        "Reply ONLY with JSON:\n"
        '{"person_approaching": bool, "is_trick_or_treater": bool, "count": int, '
        '"costumes": "<=6 words", "details": "<=15 words", "confidence": float}'
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "format": "json",
        "stream": False,
        "keep_alive": "15m",
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 90,
        },
    }

    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"Local vision inference failed ({elapsed:.2f}s): {e}")
        return {
            "person_approaching": False,
            "is_trick_or_treater": False,
            "count": 0,
            "costumes": "unknown",
            "details": f"Inference error: {e}",
            "confidence": 0.0,
            "inference_time_s": round(elapsed, 2),
        }

    elapsed = time.time() - t0
    content = res.get("message", {}).get("content", "").strip()

    # Strip optional markdown code fences
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        data = json.loads(content.strip())
        data["inference_time_s"] = round(elapsed, 2)
        return data
    except Exception as e:
        logger.warning(f"Failed to parse model JSON: {e} | Content: {content!r}")
        return {
            "person_approaching": False,
            "is_trick_or_treater": False,
            "count": 0,
            "costumes": "unknown",
            "details": content[:150],
            "confidence": 0.0,
            "inference_time_s": round(elapsed, 2),
        }


def send_discord_alert(
    token: str,
    channel_id: str,
    img_bytes: bytes,
    result: Dict[str, Any],
    is_test: bool = False,
) -> bool:
    """Send rich Halloween Discord alert with photo attachment and costume metadata."""
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {token}"}

    is_tot = result.get("is_trick_or_treater", False)
    count = result.get("count", 1)
    costumes = result.get("costumes", "Not specified")
    details = result.get("details", "Visitor approaching front door.")
    conf = int(result.get("confidence", 0.9) * 100)

    if is_test:
        title = "🎃 [TEST] Front Door Halloween Sentry Active!"
        headline = "Testing notification pipeline. System is watching for trick-or-treaters!"
    elif is_tot:
        title = f"🎃 TRICK-OR-TREATER ALERT! ({count} approaching)"
        headline = "🍬 **Trick-or-treaters detected at your front door!**"
    else:
        title = f"👀 Visitor Approaching Front Door ({count} person)"
        headline = "A visitor is approaching the porch/door."

    fields = [
        {"name": "👥 Visitors", "value": f"{count}", "inline": True},
        {"name": "🎭 Costumes / Attire", "value": f"{costumes}", "inline": True},
        {"name": "🎯 Confidence", "value": f"{conf}%", "inline": True},
        {"name": "📝 Details", "value": f"{details}", "inline": False},
    ]

    embed = {
        "title": title,
        "description": headline,
        "color": 16741656,  # Halloween Pumpkin Orange (#FF7518)
        "fields": fields,
        "image": {"url": "attachment://trick_or_treater.jpg"},
        "footer": {
            "text": f"Local AI Sentry • Gemma 4 Vision ({result.get('inference_time_s', 0)}s) • {datetime.now().strftime('%I:%M:%S %p')}"
        },
    }

    content_msg = (
        "🎃 **Ding-dong! Trick-or-treater approaching the front door!**"
        if is_tot
        else "👀 **Front door visitor detected!**"
    )

    payload = {
        "content": content_msg,
        "embeds": [embed],
    }

    clean_img = fix_color_cast(img_bytes)

    try:
        files = {
            "files[0]": ("trick_or_treater.jpg", clean_img, "image/jpeg"),
        }
        data = {"payload_json": json.dumps(payload)}

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=headers, data=data, files=files)

        if resp.status_code in (200, 201):
            logger.info("✅ Discord alert dispatched successfully!")
            return True
        else:
            logger.error(f"Discord API error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to post to Discord: {e}")
        return False


def run_sentry(
    cam_url: str,
    cam_user: str,
    cam_pass: str,
    threshold: float,
    cooldown_seconds: int,
    discord_token: str,
    discord_channel: str,
    ollama_url: str,
    model: str,
    once: bool = False,
    test_alert: bool = False,
):
    """Main execution loop for Halloween Sentry."""
    if not discord_token or not discord_channel:
        logger.error("Missing Discord credentials: DISCORD_TOKEN and DISCORD_CHANNEL_ID required.")
        sys.exit(1)

    logger.info(f"Connecting to Edge Camera at {cam_url}...")
    cam = CameraClient(cam_url, cam_user, cam_pass)

    if test_alert:
        logger.info("Triggering test alert with live camera snapshot...")
        raw = cam.fetch_frame_bytes()
        res = {
            "person_approaching": False,
            "is_trick_or_treater": True,
            "count": 1,
            "costumes": "Halloween Test Ghost 👻",
            "details": "Live camera frame verified. Alert pipeline operational!",
            "confidence": 1.0,
            "inference_time_s": 0.0,
        }
        send_discord_alert(discord_token, discord_channel, raw, res, is_test=True)
        return

    logger.info("Fetching initial baseline frame...")
    last_raw = cam.fetch_frame_bytes()
    last_downsampled = CameraClient.downsample_frame(last_raw)
    last_alert_time = 0.0

    logger.info(
        f"Sentry active! Threshold: {threshold:.1f} | Cooldown: {cooldown_seconds}s | Model: {model}"
    )

    while True:
        try:
            time.sleep(0.5)
            curr_raw = cam.fetch_frame_bytes()
            curr_downsampled = CameraClient.downsample_frame(curr_raw)

            delta = compute_frame_difference(last_downsampled, curr_downsampled)
            now = time.time()

            if delta >= threshold:
                logger.info(f"🚨 Motion detected! Pixel delta: {delta:.2f} >= {threshold:.1f}")

                if (now - last_alert_time) < cooldown_seconds:
                    remaining = int(cooldown_seconds - (now - last_alert_time))
                    logger.info(f"Cooldown active ({remaining}s remaining). Skipping inference.")
                else:
                    logger.info(f"Dispatching full frame to local vision model ({model})...")
                    eval_res = analyze_with_local_vision(
                        curr_raw, ollama_url=ollama_url, model=model
                    )
                    logger.info(
                        f"Vision Result ({eval_res.get('inference_time_s')}s): {eval_res}"
                    )

                    approaching = eval_res.get("person_approaching", False)
                    is_tot = eval_res.get("is_trick_or_treater", False)
                    conf = eval_res.get("confidence", 0.0)

                    if (approaching or is_tot) and conf >= 0.5:
                        logger.info("🎯 Visitor / Trick-or-treater confirmed! Sending alert...")
                        threading.Thread(
                            target=send_discord_alert,
                            args=(discord_token, discord_channel, curr_raw, eval_res),
                            daemon=True,
                        ).start()
                        last_alert_time = now
                    else:
                        logger.info(
                            f"No person approaching (approaching={approaching}, conf={conf:.2f})."
                        )

            # Rolling adaptive background blend: adapts in ~2s so non-moving artifacts settle
            last_downsampled = Image.blend(last_downsampled, curr_downsampled, 0.35)

            if once:
                break

        except KeyboardInterrupt:
            logger.info("Halloween sentry stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in sentry loop: {e}")
            time.sleep(2.0)


def main():
    env = load_env()

    parser = argparse.ArgumentParser(
        description="Halloween Front Door Trick-or-Treater Sentry (Local Multimodal Edge AI)"
    )
    parser.add_argument(
        "--url",
        default=os.getenv("WYZE_CAM_URL", env.get("WYZE_CAM_URL", "http://192.168.1.100/cgi-bin/currentpic.cgi")),
        help="Camera snapshot URL",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("WYZE_CAM_USER", env.get("WYZE_CAM_USER", "admin")),
        help="Camera HTTP basic auth user",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("WYZE_CAM_PASS", env.get("WYZE_CAM_PASS", "")),
        help="Camera HTTP basic auth password",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("MOTION_THRESHOLD", env.get("MOTION_THRESHOLD", "3.0"))),
        help="Pixel difference threshold for motion (default 3.0)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=int(os.getenv("ALERT_COOLDOWN_SEC", env.get("ALERT_COOLDOWN_SEC", "30"))),
        help="Cooldown between alerts in seconds (default 30)",
    )
    parser.add_argument(
        "--discord-token",
        default=os.getenv("DISCORD_TOKEN", env.get("DISCORD_TOKEN", "")),
        help="Discord bot token",
    )
    parser.add_argument(
        "--channel",
        default=os.getenv("DISCORD_CHANNEL_ID", env.get("DISCORD_CHANNEL_ID", "")),
        help="Discord channel ID",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", env.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")),
        help="Local Ollama chat endpoint",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VISION_MODEL", env.get("VISION_MODEL", "gemma4:e2b-it-q4_K_M")),
        help="Local vision model tag",
    )
    parser.add_argument("--once", action="store_true", help="Run a single pass")
    parser.add_argument("--test-alert", action="store_true", help="Send a test alert with current frame")

    args = parser.parse_args()
    run_sentry(
        cam_url=args.url,
        cam_user=args.user,
        cam_pass=args.password,
        threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        discord_token=args.discord_token,
        discord_channel=args.channel,
        ollama_url=args.ollama_url,
        model=args.model,
        once=args.once,
        test_alert=args.test_alert,
    )


if __name__ == "__main__":
    main()

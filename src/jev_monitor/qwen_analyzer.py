"""Analyze candidate incidents with one locally loaded Qwen2.5-Omni model."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol


MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
REQUIRED_ANALYSIS_KEYS = {
    "is_anomaly",
    "anomaly_type",
    "severity",
    "summary",
    "confidence",
    "evidence",
    "recommended_action",
}
SEVERITIES = {"none", "low", "medium", "high", "critical"}

ANALYSIS_PROMPT = """You are the second-stage reviewer in a home camera anomaly system.
Analyze the short incident clip using all available visual and audio evidence.
Normal pet behavior includes resting, walking, playing, barking, howling, approaching
the camera, and responding to an owner's voice when there is no harm or hazard.
An anomaly includes injury, a fall, severe distress, intrusion, smoke, fire,
contamination such as feces or vomit, property damage, or another household hazard.
Do not call ordinary motion, camera proximity, low light, or harmless vocalization an
anomaly by itself. Return exactly one JSON object with these fields:
{
  "is_anomaly": boolean,
  "anomaly_type": string,
  "severity": "none" | "low" | "medium" | "high" | "critical",
  "summary": string,
  "confidence": number from 0 to 1,
  "evidence": [string],
  "recommended_action": string
}
Use anomaly_type "none" and severity "none" when the incident is normal.
Do not use Markdown fences or add text outside the JSON object."""


class ClipAnalyzer(Protocol):
    model_id: str

    def analyze(self, clip_path: Path, use_audio: bool) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ClipBounds:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def incident_clip_bounds(
    incident: dict[str, Any],
    pre_roll_seconds: float,
    post_roll_seconds: float,
    max_clip_seconds: float,
) -> ClipBounds:
    if pre_roll_seconds < 0 or post_roll_seconds < 0:
        raise ValueError("Clip pre-roll and post-roll cannot be negative")
    if max_clip_seconds <= 0:
        raise ValueError("Maximum clip duration must be greater than zero")
    start = max(0.0, float(incident["started_at_seconds"]) - pre_roll_seconds)
    end = max(start + 0.1, float(incident["ended_at_seconds"]) + post_roll_seconds)
    if end - start > max_clip_seconds:
        peak_center = float(incident["peak_window_start_seconds"]) + 0.5
        start = max(0.0, peak_center - max_clip_seconds / 2)
        end = start + max_clip_seconds
    return ClipBounds(round(start, 3), round(end, 3))


def extract_clip(video_path: Path, output_path: Path, bounds: ClipBounds) -> None:
    ffmpeg = ffmpeg_executable()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(bounds.start),
        "-i",
        str(video_path),
        "-t",
        str(bounds.duration),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "ffmpeg is required; install the qwen extra or provide ffmpeg on PATH"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_has_audio(video_path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    result = subprocess.run(
        [ffmpeg_executable(), "-hide_banner", "-i", str(video_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return "Audio:" in result.stderr


def parse_json_response(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and REQUIRED_ANALYSIS_KEYS <= value.keys():
            return validate_analysis(value)
    raise ValueError(f"Qwen did not return the required JSON object: {text[:500]!r}")


def validate_analysis(value: dict[str, Any]) -> dict[str, Any]:
    severity = str(value["severity"]).lower()
    if severity not in SEVERITIES:
        raise ValueError(f"Unsupported severity from Qwen: {severity}")
    confidence = float(value["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("Qwen confidence must be between 0 and 1")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ValueError("Qwen evidence must be a list of strings")
    if not isinstance(value["is_anomaly"], bool):
        raise ValueError("Qwen is_anomaly must be a boolean")
    is_anomaly = value["is_anomaly"]
    if not is_anomaly:
        severity = "none"
    return {
        "is_anomaly": is_anomaly,
        "anomaly_type": str(value["anomaly_type"]),
        "severity": severity,
        "summary": str(value["summary"]),
        "confidence": confidence,
        "evidence": evidence,
        "recommended_action": str(value["recommended_action"]),
    }


def action_for_analysis(analysis: dict[str, Any]) -> str:
    if not analysis["is_anomaly"]:
        return "log_only"
    return {
        "none": "log_only",
        "low": "log_only",
        "medium": "queue_review",
        "high": "notify",
        "critical": "alert_immediately",
    }[analysis["severity"]]


class QwenOmniAnalyzer:
    def __init__(self, model_id: str = MODEL_ID, attn_implementation: str | None = None) -> None:
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen2.5-Omni analysis requires a CUDA GPU")
        model_options: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            model_options["attn_implementation"] = attn_implementation
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            **model_options,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.process_mm_info = process_mm_info
        self.torch = torch
        self.model_id = model_id

    def analyze(self, clip_path: Path, use_audio: bool) -> dict[str, Any]:
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a precise home-camera incident reviewer.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(clip_path)},
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = self.process_mm_info(
            conversation,
            use_audio_in_video=use_audio,
        )
        inputs = self.processor(
            text=prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                return_audio=False,
                thinker_do_sample=False,
                thinker_max_new_tokens=512,
            )
        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_json_response(response)


def analyze_incident_file(
    incident_path: Path,
    video_dir: Path,
    output_path: Path,
    analyzer: ClipAnalyzer,
    pre_roll_seconds: float = 1.0,
    post_roll_seconds: float = 1.0,
    max_clip_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    incidents = [
        json.loads(line)
        for line in incident_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="jevguard-qwen-") as directory:
        temporary_dir = Path(directory)
        for incident in incidents:
            camera_id = str(incident["camera_id"])
            video_path = video_dir / f"{camera_id}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"Video for {camera_id} not found: {video_path}")
            bounds = incident_clip_bounds(
                incident,
                pre_roll_seconds,
                post_roll_seconds,
                max_clip_seconds,
            )
            clip_path = temporary_dir / f"{incident['incident_id']}.mp4"
            extract_clip(video_path, clip_path, bounds)
            use_audio = video_has_audio(clip_path)
            analysis = analyzer.analyze(clip_path, use_audio=use_audio)
            analyzed = {
                **incident,
                "action": action_for_analysis(analysis),
                "analysis": {**analysis, "provider": analyzer.model_id},
                "status": "analyzed",
                "media": {
                    "video_path": str(video_path),
                    "clip_start_seconds": bounds.start,
                    "clip_end_seconds": bounds.end,
                    "audio_used": use_audio,
                },
            }
            output.append(analyzed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for incident in output:
            handle.write(json.dumps(incident, ensure_ascii=True) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze JevGuard incidents with Qwen2.5-Omni")
    parser.add_argument("--incidents-dir", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--pre-roll-seconds", type=float, default=1.0)
    parser.add_argument("--post-roll-seconds", type=float, default=1.0)
    parser.add_argument("--max-clip-seconds", type=float, default=8.0)
    parser.add_argument("--attn-implementation")
    args = parser.parse_args()

    incident_paths = sorted(
        path for path in args.incidents_dir.glob("*.jsonl") if path.name != "summary.jsonl"
    )
    if not incident_paths:
        raise SystemExit(f"No incident JSONL files found in {args.incidents_dir}")

    analyzer = QwenOmniAnalyzer(args.model_id, args.attn_implementation)
    total = 0
    videos = []
    for path in incident_paths:
        analyzed = analyze_incident_file(
            path,
            args.video_dir,
            args.output_dir / path.name,
            analyzer,
            args.pre_roll_seconds,
            args.post_roll_seconds,
            args.max_clip_seconds,
        )
        total += len(analyzed)
        videos.append({"video_id": path.stem, "incidents": len(analyzed)})

    summary = {
        "model": analyzer.model_id,
        "total_incidents": total,
        "videos": videos,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

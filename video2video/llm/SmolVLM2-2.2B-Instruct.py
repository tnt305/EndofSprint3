import os
import json
import gradio as gr
import torch
import tempfile
from pathlib import Path
import subprocess
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from transformers import AutoProcessor, AutoModelForImageTextToText


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_examples(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)

def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_video_duration_seconds(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])

class VideoHighlightDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 8
    ):
        self.device = device
        self.batch_size = batch_size
        
        # Initialize model and processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
        
    def analyze_video_content(self, video_path: str) -> str:
        system_message = "You are a helpful assistant that can understand videos. Describe what type of video this is and what's happening in it."
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": "What type of video is this and what's happening in it? Be specific about the content type and general activities you observe."}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        return self.processor.decode(outputs[0], skip_special_tokens=True).lower().split("assistant: ")[1]

    def analyze_segment(self, video_path: str) -> str:
        """Analyze a specific video segment and provide a brief description."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Focus only on describing the key dramatic action or notable event occurring in this video segment. Skip general context or scene-setting details unless they are crucial to understanding the main action."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": "WWhat is the main action or notable event happening in this segment? Describe it in one brief sentence."}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        return self.processor.decode(outputs[0], skip_special_tokens=True).split("Assistant: ")[1]

    def determine_highlights(self, video_description: str, prompt_num: int = 1) -> str:
        """Determine what constitutes highlights based on video description with different prompts."""
        system_prompts = {
            1: "You are a highlight editor. List archetypal dramatic moments that would make compelling highlights if they appear in the video. Each moment should be specific enough to be recognizable but generic enough to potentially exist in other videos of this type.",
            2: "You are a helpful visual-language assistant that can understand videos and edit. You are tasked helping the user to create highlight reels for videos. Highlights should be rare and important events in the video in question."
        }
        user_prompts = {
            1: "List potential highlight moments to look for in this video:",
            2: "List dramatic moments that would make compelling highlights if they appear in the video. Each moment should be specific enough to be recognizable but generic enough to potentially exist in any video of this type:"
        }
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompts[prompt_num]}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"""Here is a description of a video:\n\n{video_description}\n\n{user_prompts[prompt_num]}"""}]
            }
        ]

        print(f"Using prompt {prompt_num} for highlight detection")
        print(messages)
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        return self.processor.decode(outputs[0], skip_special_tokens=True).split("Assistant: ")[1]

    def process_segment(self, video_path: str, highlight_types: str) -> bool:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a video highlight analyzer. Your role is to identify moments that have high dramatic value, focusing on displays of skill, emotion, personality, or tension. Compare video segments against provided example highlights to find moments with similar emotional impact and visual interest, even if the specific actions differ."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": f"""Given these highlight examples:\n{highlight_types}\n\nDoes this video contain a moment that matches the core action of one of the highlights? Answer with:\n'yes' or 'no'\nIf yes, justify it"""}]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = self.processor.decode(outputs[0], skip_special_tokens=True).lower().split("assistant: ")[1]
        return "yes" in response

def create_xspf_playlist(video_path: str, segments: list, descriptions: list) -> str:
    """Create XSPF playlist from segments with descriptions."""
    # Get video filename with full path
    video_filename = os.path.basename(video_path)
    
    # Create the XML structure as a string
    xml_content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<playlist version="1" xmlns="http://xspf.org/ns/0/" xmlns:vlc="http://www.videolan.org/vlc/playlist/0/">',
        f'  <title>{video_filename} - Highlights</title>',
        '  <trackList>'
    ]
    
    for idx, ((start_time, end_time), description) in enumerate(zip(segments, descriptions)):
        track = [
            '    <track>',
            f'      <location>file:///{video_filename}</location>',
            f'      <title>{description}</title>',
            f'      <annotation>{description}</annotation>',
            '      <extension application="http://www.videolan.org/vlc/playlist/0">',
            f'        <vlc:id>{idx}</vlc:id>',
            f'        <vlc:option>start-time={int(start_time)}</vlc:option>',
            f'        <vlc:option>stop-time={int(end_time)}</vlc:option>',
            '      </extension>',
            '    </track>'
        ]
        xml_content.extend(track)
    
    xml_content.extend([
        '  </trackList>',
        '</playlist>'
    ])
    
    return '\n'.join(xml_content)


model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
                
def on_process(video):
    duration = get_video_duration_seconds(video)
    detector = VideoHighlightDetector(model_path=model_path, batch_size=16)

    # Analyze video content
    video_desc = detector.analyze_video_content(video)
    formatted_desc = f"### Video Summary:\n{video_desc}"

    highlights1 = detector.determine_highlights(video_desc, prompt_num=1)
    highlights2 = detector.determine_highlights(video_desc, prompt_num=2)
    formatted_highlights = f"### Highlight Criteria:\nSet 1:\n{highlights1}\n\nSet 2:\n{highlights2}"
        
    # Process video in segments
    segment_length = 10.0
    kept_segments1 = []
    kept_segments2 = []
    segment_descriptions1 = []
    segment_descriptions2 = []
    segments_processed = 0
    total_segments = int(duration / segment_length)

    for start_time in range(0, int(duration), int(segment_length)):
        end_time = min(start_time + segment_length, duration)
        
        # Create temporary segment
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_segment:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video,
                "-ss", str(start_time),
                "-t", str(segment_length),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                temp_segment.name
            ]
            subprocess.run(cmd, check=True)
            
            # Process with both highlight sets
            if detector.process_segment(temp_segment.name, highlights1):
                description = detector.analyze_segment(temp_segment.name)
                kept_segments1.append((start_time, end_time))
                segment_descriptions1.append(description)
                
            if detector.process_segment(temp_segment.name, highlights2):
                description = detector.analyze_segment(temp_segment.name)
                kept_segments2.append((start_time, end_time))
                segment_descriptions2.append(description)

        segments_processed += 1

    # Calculate percentages of video kept for each highlight set
    total_duration = duration
    duration1 = sum(end - start for start, end in kept_segments1)
    duration2 = sum(end - start for start, end in kept_segments2)
    
    percent1 = (duration1 / total_duration) * 100
    percent2 = (duration2 / total_duration) * 100
    
    print(f"Highlight set 1: {percent1:.1f}% of video")
    print(f"Highlight set 2: {percent2:.1f}% of video")

    # Choose the set with lower percentage unless it's zero
    if (0 < percent2 <= percent1 or percent1 == 0):
        final_segments = kept_segments2
        segment_descriptions = segment_descriptions2
        selected_set = "2"
        percent_used = percent2
    else:
        final_segments = kept_segments1
        segment_descriptions = segment_descriptions1
        selected_set = "1"
        percent_used = percent1

    if final_segments:
        # Create XSPF playlist
        playlist_content = create_xspf_playlist(video, final_segments, segment_descriptions)
        
        # Save playlist to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xspf', delete=False) as f:
            f.write(playlist_content)
            playlist_path = f.name
        
        completion_message = f"Processing complete! Using highlight set {selected_set} ({percent_used:.1f}% of video). You can download the playlist."
        
    torch.cuda.empty_cache()


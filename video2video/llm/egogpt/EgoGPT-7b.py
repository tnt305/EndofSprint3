import os
import re
import sys
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from typing import Optional

import threading
from transformers import TextIteratorStreamer


# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
import uvicorn
import librosa
import whisper
import requests
from pydantic import BaseModel
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from datetime import datetime
import shutil

# Local imports
from egogpt.model.builder import load_pretrained_model
from egogpt.mm_utils import get_model_name_from_path, process_images
from egogpt.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN, 
    IGNORE_INDEX,
    SPEECH_TOKEN_INDEX,
    DEFAULT_SPEECH_TOKEN
)
from egogpt.conversation import conv_templates, SeparatorStyle
import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download

# Download the model checkpoint file (large-v3.pt)
ego_gpt_path = hf_hub_download(
    repo_id="lmms-lab/EgoGPT-7b-Demo",
    filename="speech_encoder/large-v3.pt",
    local_dir="./",
)

import shutil

try:
    os.chmod("./", 0o777)
    shutil.move('./speech_encoder/large-v3.pt', '/large-v3.pt')
except PermissionError as e:
    subprocess.run(['mv', './speech_encoder/large-v3.pt', './large-v3.pt'])




pretrained = "lmms-lab/EgoGPT-7b-Demo"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_map = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add this initialization code before loading the model
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12377'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

setup(0,1)
tokenizer, model, max_length = load_pretrained_model(pretrained,device_map=device_map)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()


# cur_dir = os.path.dirname(os.path.abspath(__file__))
cur_dir = '.'
# Add this after cur_dir definition
UPLOADS_DIR = os.path.join(cur_dir, "user_uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def time_to_frame_idx(time_int: int, fps: int) -> int:
    """
    Convert time in HHMMSSFF format (integer or string) to frame index.
    :param time_int: Time in HHMMSSFF format, e.g., 10483000 (10:48:30.00) or "10483000".
    :param fps: Frames per second of the video.
    :return: Frame index corresponding to the given time.
    """
    # Ensure time_int is a string for slicing
    time_str = str(time_int).zfill(
        8)  # Pad with zeros if necessary to ensure it's 8 digits

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    frames = int(time_str[6:8])

    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_frames = total_seconds * fps + frames  # Convert to total frames

    return total_frames

def split_text(text, keywords):
    # 创建一个正则表达式模式，将所有关键词用 | 连接，并使用捕获组
    pattern = '(' + '|'.join(map(re.escape, keywords)) + ')'
    # 使用 re.split 保留分隔符
    parts = re.split(pattern, text)
    # 去除空字符串
    parts = [part for part in parts if part]
    return parts

warnings.filterwarnings("ignore")

# Create FastAPI instance
app = FastAPI()
def load_video(
    video_path: Optional[str] = None,
    max_frames_num: int = 16,
    fps: int = 1,
    video_start_time: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    time_based_processing: bool = False
) -> tuple:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    target_sr = 16000
    
    # Process video frames first
    if time_based_processing:
        # Initialize video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()
        
        # Convert time to frame index based on the actual video FPS
        video_start_frame = int(time_to_frame_idx(video_start_time, video_fps))
        start_frame = int(time_to_frame_idx(start_time, video_fps))
        end_frame = int(time_to_frame_idx(end_time, video_fps))

        print("start frame", start_frame)
        print("end frame", end_frame)

        # Ensure the end time does not exceed the total frame number
        if end_frame - start_frame > total_frame_num:
            end_frame = total_frame_num + start_frame

        # Adjust start_frame and end_frame based on video start time
        start_frame -= video_start_frame
        end_frame -= video_start_frame
        start_frame = max(0, int(round(start_frame)))  # 确保不会小于0
        end_frame = min(total_frame_num, int(round(end_frame))) # 确保不会超过总帧数
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        # Sample frames based on the provided fps (e.g., 1 frame per second)
        frame_idx = [i for i in range(start_frame, end_frame) if (i - start_frame) % int(video_fps / fps) == 0]

        # Get the video frames for the sampled indices
        video = vr.get_batch(frame_idx).asnumpy()
    else:
        # Original video processing logic
        total_frame_num = len(vr)
        avg_fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
        
        if max_frames_num > 0:
            if len(frame_idx) > max_frames_num:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
        
        video = vr.get_batch(frame_idx).asnumpy()

    # Try to load audio, return None for speech if failed
    try:
        if time_based_processing:
            y, _ = librosa.load(video_path, sr=target_sr)
            start_sample = int(start_time * target_sr)
            end_sample = int(end_time * target_sr)
            speech = y[start_sample:end_sample]
        else:
            speech, _ = librosa.load(video_path, sr=target_sr)
            
        # Process audio if it exists
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
        
        return video, speech, speech_lengths, True  # True indicates real audio
        
    except Exception as e:
        print(f"Warning: Could not load audio from video: {e}")
        # Create dummy silent audio
        duration = 10  # 10 seconds
        speech = np.zeros(duration * target_sr, dtype=np.float32)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
        return video, speech, speech_lengths, False  # False indicates no real audio

class PromptRequest(BaseModel):
    prompt: str
    video_path: str = None
    max_frames_num: int = 16
    fps: int = 1
    video_start_time: float = None
    start_time: float = None
    end_time: float = None
    time_based_processing: bool = False

# @spaces.GPU(duration=120)
def save_interaction(video_path, prompt, output, audio_path=None):
    """Save user interaction data and files"""
    if not video_path:
        return
    
    # Create timestamped directory for this interaction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interaction_dir = os.path.join(UPLOADS_DIR, timestamp)
    os.makedirs(interaction_dir, exist_ok=True)
    
    # Copy video file
    video_ext = os.path.splitext(video_path)[1]
    new_video_path = os.path.join(interaction_dir, f"video{video_ext}")
    shutil.copy2(video_path, new_video_path)
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "prompt": prompt,
        "output": output,
        "video_path": new_video_path,
    }
    
    # Only try to save audio if it's a file path (str), not audio data (tuple)
    if audio_path and isinstance(audio_path, (str, bytes, os.PathLike)):
        audio_ext = os.path.splitext(audio_path)[1]
        new_audio_path = os.path.join(interaction_dir, f"audio{audio_ext}")
        shutil.copy2(audio_path, new_audio_path)
        metadata["audio_path"] = new_audio_path
    
    with open(os.path.join(interaction_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def extract_audio_from_video(video_path, audio_path=None):
    print('Processing audio from video...', video_path, audio_path)
    if video_path is None:
        return None
        
    if isinstance(video_path, dict) and 'name' in video_path:
        video_path = video_path['name']
    
    try:
        y, sr = librosa.load(video_path, sr=8000, mono=True, res_type='kaiser_fast')
        # Check if the audio is silent
        if np.abs(y).mean() < 0.001:
            print("Video appears to be silent")
            return None
        return (sr, y)
    except Exception as e:
        print(f"Warning: Could not extract audio from video: {e}")
        return None

import time

@spaces.GPU
def generate_text(video_path, audio_track, prompt):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    max_frames_num = 30
    fps = 1
    conv_template = "qwen_1_5"
    if video_path is None and audio_track is None:
        question = prompt
        speech = None
        speech_lengths = None
        has_real_audio = False
        image = None
        image_sizes= None
        modalities = ["image"]
        image_tensor=None
    # Load video and potentially audio
    else:
        video, speech, speech_lengths, has_real_audio = load_video(
            video_path=video_path,
            max_frames_num=max_frames_num,
            fps=fps,
        )

        # Prepare the prompt based on whether we have real audio
        if not has_real_audio:
            question = f"<image>\n{prompt}"  # Video-only prompt
        else:
            question = f"<speech>\n<image>\n{prompt}"  # Video + speech prompt

        speech = torch.stack([speech]).to("cuda").half()
        processor = model.get_vision_tower().image_processor
        processed_video = processor.preprocess(video, return_tensors="pt")["pixel_values"]
        image = [(processed_video, video[0].size, "video")]
        image_tensor = [image[0][0].half()]
        image_sizes = [image[0][1]]
        modalities = ["video"]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()



    parts = split_text(prompt_question, ["<image>", "<speech>"])
    input_ids = []
    for part in parts:
        if "<image>" == part:
            input_ids += [IMAGE_TOKEN_INDEX]
        elif "<speech>" == part and speech is not None:  # Only add speech token if we have audio
            input_ids += [SPEECH_TOKEN_INDEX]
        else:
            input_ids += tokenizer(part).input_ids

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)


    generate_kwargs = {"eos_token_id": tokenizer.eos_token_id}

    def generate_response():
        model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            speech=speech,
            speech_lengths=speech_lengths,
            do_sample=False,
            temperature=0.7,
            max_new_tokens=512,
            repetition_penalty=1.2,
            modalities=modalities,
            streamer=streamer,
            **generate_kwargs
        )

    # Start generation in a separate thread
    thread = threading.Thread(target=generate_response)
    thread.start()

    # Stream the output word by word
    generated_text = ""
    partial_word = ""
    cursor = "|"  
    cursor_visible = True
    last_cursor_toggle = time.time()

    for new_text in streamer:
        partial_word += new_text
        # Toggle the cursor visibility every 0.5 seconds
        if time.time() - last_cursor_toggle > 0.5:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()
        current_cursor = cursor if cursor_visible else " "
        if partial_word.endswith(" ") or partial_word.endswith("\n"):
            generated_text += partial_word
            # Yield the current text with the cursor appended
            yield generated_text + current_cursor
            partial_word = ""
        else:
            # Yield the current text plus the partial word and the cursor
            yield generated_text + partial_word + current_cursor

    # Handle any remaining partial word at the end
    if partial_word:
        generated_text += partial_word
        yield generated_text

    # Save the interaction after generation is complete
    save_interaction(video_path, prompt, generated_text, audio_track)
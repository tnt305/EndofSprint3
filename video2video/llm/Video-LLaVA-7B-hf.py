import gradio as gr
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, TextIteratorStreamer
from threading import Thread
import re
import time 
from PIL import Image
import torch
import cv2

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="cuda")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
#model.to("cuda")

def replace_video_with_images(text, frames):
  return text.replace("<video>", "<image>" * frames)

import cv2
from PIL import Image

def sample_frames(video_file, num_frames):
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)
    frames = []
    
    for i in range(0, total_frames, interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(pil_img)
        if len(frames) == num_frames:
            break

    video.release()
    return frames

def bot_streaming(message, history):

  txt = message.text
  ext_buffer = f"USER: {txt} ASSISTANT: "

  if message.files:
    if len(message.files) == 1:
      image = [message.files[0].path]
    # interleaved images or video
    elif len(message.files) > 1:
      image = [msg.path for msg in message.files]
  else:
      
    def has_file_data(lst):
      return any(isinstance(item, FileData) for sublist in lst if isinstance(sublist, tuple) for item in sublist)

    def extract_paths(lst):
        return [item.path for sublist in lst if isinstance(sublist, tuple) for item in sublist if isinstance(item, FileData)]

    latest_text_only_index = -1

    for i, item in enumerate(history):
        if all(isinstance(sub_item, str) for sub_item in item):
            latest_text_only_index = i

    image = [path for i, item in enumerate(history) if i < latest_text_only_index and has_file_data(item) for path in extract_paths(item)]
      
  video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg")
  image_extensions = Image.registered_extensions()
  image_extensions = tuple([ex for ex, f in image_extensions.items()])
  image_list = []
  video_list = []

  print("media", image)
  if len(image) == 1:
    if image[0].endswith(video_extensions):
        
        video_list = sample_frames(image[0], 12)
        
        prompt = f"USER: <video> {message.text} ASSISTANT:"
    elif image[0].endswith(image_extensions):
        image_list.append(Image.open(image[0]).convert("RGB"))
        prompt =  f"USER: <image> {message.text} ASSISTANT:"

  elif len(image) > 1:
    user_prompt = message.text

    for img in image:
      if img.endswith(image_extensions):
        img = Image.open(img).convert("RGB")
        image_list.append(img)

      elif img.endswith(video_extensions):
        video_list.append(sample_frames(img, 7))
        print(len(video_list[-1]))
        #for frame in sample_frames(img, 6):
          #video_list.append(frame)
        
    print("video_list", video_list)
    image_tokens = ""
    video_tokens = ""

    if image_list != []:
      image_tokens = "<image>" * len(image_list)
    if video_list != []:   
      
      toks = len(video_list) 
      video_tokens = "<video>" * toks
      
    

    prompt = f"USER: {image_tokens}{video_tokens} {user_prompt} ASSISTANT:"

  print(prompt)
  if image_list != [] and video_list != []:
    inputs = processor(prompt, images=image_list, videos=video_list, return_tensors="pt").to("cuda",torch.float16)
  elif image_list != [] and video_list == []:
    inputs = processor(prompt, images=image_list, return_tensors="pt").to("cuda", torch.float16)
  elif image_list == [] and video_list != []:
    inputs = processor(prompt, videos=video_list, return_tensors="pt").to("cuda", torch.float16)
  
  
  streamer = TextIteratorStreamer(processor, **{"max_new_tokens": 200, "skip_special_tokens": True, "clean_up_tokenization_spaces":True})
  generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
  generated_text = ""

  thread = Thread(target=model.generate, kwargs=generation_kwargs)
  thread.start()

  

  buffer = ""
  for new_text in streamer:
    
    buffer += new_text
    
    generated_text_without_prompt = buffer[len(ext_buffer):][:-1]
    time.sleep(0.01)
    yield generated_text_without_prompt
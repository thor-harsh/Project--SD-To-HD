
# Video Conversion Prototype

This project converts SD resolution videos (640x480) to HD resolution videos (1280x720) using a diffusion model. The tool fills in the blank areas on the sides while preserving the video context.

## Installation

Install the required libraries with:

```bash
pip install opencv-python-headless torch diffusers transformers pillow tqdm
```

## Usage

### 1. Create a Demo Video

Generates a simple demo video.

```python
import cv2
import numpy as np

def create_demovideo(file_path, width=640, height=480, num_frames=10, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.putText(frame, str(i + 1), (width // 2 - 20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
        video.write(frame)
    video.release()

input_video_path = 'simple_demo_sd_video.mp4'
create_demovideo(input_video_path)
```

### 2. Convert Video

Converts the demo video to HD resolution.

```python
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        retur, frame = cap.read()
        if not retur:
            break
        frames.append(frame)
    cap.release()
    return frames

model_name = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')

def inpaint_and_resize(image, pipe):
    resized_image = cv2.resize(image, (512, 512))
    pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    mask = Image.new('L', (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([(0, 0), (128, 512)], fill=255)
    draw.rectangle([(384, 0), (512, 512)], fill=255)
    inpainted_image = pipe(prompt="", image=pil_image, mask_image=mask).images[0]
    inpainted_image_cv = cv2.cvtColor(np.array(inpainted_image), cv2.COLOR_RGB2BGR)
    center_original = resized_image[:, 128:384]
    inpainted_image_cv[:, 128:384] = center_original
    hd_image = cv2.resize(inpainted_image_cv, (1280, 720))
    return hd_image

def process_frame(frame, pipe):
    return inpaint_and_resize(frame, pipe)

def process_frames(frames, pipe, number_of_workers=4):
    with ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        processed_frames = list(tqdm(executor.map(lambda frame: process_frame(frame, pipe), frames), total=len(frames)))
    return processed_frames

def savevideo(frames, output_path, fps):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def convertvideo(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames = extract_frames(input_video_path)
    batch_size = 2
    processed_frames = []
    for i in tqdm(range(0, len(frames), batch_size), desc="Processing Batches"):
        batch = frames[i:i + batch_size]
        processed_batch = process_frames(batch, pipe)
        processed_frames.extend(processed_batch)
    savevideo(processed_frames, output_video_path, fps)

if __name__ == "__main__":
    input_video_path = 'simple_demo_sd_video.mp4'
    output_video_path = 'output_hd_demo_video.mp4'
    convertvideo(input_video_path, output_video_path)
```

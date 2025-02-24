import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


dataset_path = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\DADA\DADA2000\images"
videos_paths = list(Path(dataset_path).iterdir())
destination_path = r"E:\videos"

fps = 30

for path in videos_paths:  

    if Path(f"{destination_path}/{path.name}.mp4").exists():
        continue

    frames_paths = list(path.glob('*.png'))
    annotated_frames = []

    for i, frame_path in enumerate(frames_paths):
       img = Image.open(frame_path).convert("RGBA")
       draw = ImageDraw.Draw(img)
       
       text = f"Klatka: {i + 1}"
       draw.text((10, 10), text, font=ImageFont.truetype(font='arial', size=24),fill="white")
        
       annotated_frames.append(np.array(img))

       del img
    
    clip = ImageSequenceClip([frame for frame in annotated_frames], fps=fps)
    clip.write_videofile(f"{destination_path}/{path.name}.mp4", codec="libx264")

    del annotated_frames
    del clip
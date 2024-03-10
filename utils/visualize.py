import cv2
import numpy as np

def frames2video(frames, video_path, fps=30):
    '''
    frames: list of frames, each frame is a numpy array with shape (H,W,3)
    '''
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()
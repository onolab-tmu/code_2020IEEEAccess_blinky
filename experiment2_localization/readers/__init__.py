'''
Implements some readers for video and audio streams based on opencv and ffmpeg

Code by Robin Scheibler and Daiki Horiike, 2018
'''
from .video import ThreadedVideoStream, video_stream, frame_grabber
from .audio import audioread
from .processors import OnlineStats, PixelCatcher, BoxCatcher, ProcessorBase
from .ffmpeg_raw import ffmpeg_open_raw_video, ffmpeg_open_raw_audio

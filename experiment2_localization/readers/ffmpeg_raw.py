import numpy as np
import platform
import subprocess as sp

# infer platform
if platform.system() in ['Darwin', 'Linux']:
    FFMPEG_BIN = 'ffmpeg'
elif platform.system() in ['Windows']:
    FFMPEG_BIN = 'ffmpeg.exe'
else:
    print('Unknown OS, trying my luck...')
    FFMPEG_BIN = 'ffmpeg'

def _numpy_byte_count(dtype):
    ''' Shortcut to get the number of bytes of a numpy type '''
    return np.array(1, dtype=dtype).nbytes

def _read_byte_stream(command, framesize, dtype, bufsize, readsize, online_func=None, keep_data=True):
    '''
    Reads an audio file into memory using ffmpeg with optional online processing

    Parameters
    ---------
    command: list of str
        the command opening the stream using subprocess and ffmpeg
    framesize: tuple of int
        the data frame size
    dtype: numpy.type
        the data type of individual samples
    bufsize: int
        the number of frames to buffer
    readsize: int
        the number of frames to read at once
    online_func: function, optional
        a function to apply to each new batch of data read
    keep_data: bool, optional
        if True returns the raw video as a numpy array
    '''

    frame_byte_size = int(np.prod(framesize)) * _numpy_byte_count(dtype)
    bufsize_bytes = frame_byte_size * bufsize
    readsize_bytes = frame_byte_size * readsize

    showed_frame = False

    with sp.Popen(command, stdout=sp.PIPE, bufsize=bufsize_bytes) as pipe:
        chunks = []
        while True:
            raw = pipe.stdout.read(readsize_bytes)
            if len(raw) == 0:
                break
            chunk = np.fromstring(raw, dtype=dtype).reshape((-1,) + framesize)

            # apply a processing function on the streaming data
            if online_func is not None:
                for f in chunk:
                    online_func(f)

            if keep_data:
                chunks.append(chunk)

        if keep_data:
            return np.concatenate(chunks)
        else:
            return None

def ffmpeg_open_raw_audio(filename, n_channels=2, dtype=np.int16, bufsize=1, readsize=1, online_func=None, keep_data=True, 
                                    start=None, duration=None):
    '''
    Reads an audio file into memory using ffmpeg with optional online processing

    Parameters
    ---------
    n_channels: int, optional
        the number of channels in the video (default to 2 for stereo)
    dtype: numpy.type, optional
        the data type of individual samples (default np.int16)
    bufsize: int, optional
        the number of frames to buffer
    readsize: int, optional
        the number of frames to read at once
    online_func: function, optional
        a function to apply to each new batch of data read
    keep_data: bool, optional
        if True returns the raw video as a numpy array
    '''

    pcm_format = {np.int16: 's16le', np.int32: 's32le'}

    command_a = [ FFMPEG_BIN, '-i', filename, '-vn', '-f', pcm_format[dtype] ]

    if start is not None:
        command_a += [ '-ss', str(start) ]

    if duration is not None:
        command_a += ['-t', str(duration) ]

    # output to pipe
    command_a += [ '-' ]

    framesize = (n_channels,)

    return _read_byte_stream(command_a, (n_channels,), dtype, bufsize, readsize, 
                                        online_func=online_func, keep_data=keep_data)


def ffmpeg_open_raw_video(filename, framesize, n_channels=3, dtype=np.uint8, bufsize=1, readsize=1, online_func=None, keep_data=True,
                                    start=None, duration=None):
    '''
    Reads a video file into memory using ffmpeg with optional online processing

    Parameters
    ---------
    framesize: tuple of size 2
        video frame size in pixels
    n_channels: int, optional
        the number of channels in the video (default to 3 for RGB)
    dtype: numpy.type, optional
        the data type of individual samples (default np.uint8 for videos)
    bufsize: int, optional
        the number of frames to buffer
    readsize: int, optional
        the number of frames to read at once
    online_func: function, optional
        a function to apply to each new batch of data read
    keep_data: bool, optional
        if True returns the raw video as a numpy array
    '''

    frame_byte_size = int(np.prod(framesize)) * n_channels * _numpy_byte_count(dtype)
    bufsize_bytes = frame_byte_size * bufsize
    readsize_bytes = frame_byte_size * readsize

    command_v = [ FFMPEG_BIN, 
            '-i', filename, 
            '-f', 'image2pipe', 
            '-pix_fmt', 'rgb24', 
            '-vcodec', 'rawvideo',
            '-color_trc', 'linear',
            ]

    if start is not None:
        command_v += [ '-ss', str(start) ]

    if duration is not None:
        command_v += ['-t', str(duration) ]

    # output to pipe
    command_v += [ '-' ]

    return _read_byte_stream(command_v, framesize[::-1] + (n_channels,), dtype, bufsize, readsize,
                                        online_func=online_func, keep_data=keep_data)

import argparse, json, os
import jsongzip
import numpy as np
import cv2
import chainer
import matplotlib.pyplot as plt
from scipy.io import wavfile
from readers import ThreadedVideoStream, ProcessorBase, PixelCatcher, BoxCatcher
from ml_localization import get_data_raw, get_data, models, get_formatters

TEMP_VIDEO_FILE = 'temp.avi'

if __name__ == '__main__':

    video_choices = [ 'noise', 'speech', 'hori_1', 'hori_2', 'hori_3', 'hori_4', 'hori_5', ]
    path_choices = ['diagonal', 'parallel_long', 'parallel_short']

    parser = argparse.ArgumentParser(description='Show the estimated source location on top of the video')
    parser.add_argument('protocol', type=str,
            help='The protocol file containing the experiment metadata')
    parser.add_argument('config', type=str, help='The JSON file containing the configuration.')
    parser.add_argument('-v', '--video', type=str, choices=video_choices,
            default=video_choices[0], help='The video segment to process')
    parser.add_argument('-p', '--path', type=str, choices=path_choices,
            default=path_choices[0], help='The path to process')
    parser.add_argument('-s', '--save', type=str, metavar='PATH',
            help='Save the video to given path')
    parser.add_argument('--no_show', action='store_true',
            help='Do not show the video while processing')
    parser.add_argument('-f', '--format', choices=['mov','mp4','avi'], default='avi')
    args = parser.parse_args()

    # get the path to the experiment files
    experiment_path = os.path.split(args.protocol)[0]

    # Read some parameters from the protocol
    with open(args.protocol,'r') as f:
        protocol = json.load(f)

    # read in the important stuff
    blinkies = np.array(protocol['blinky_locations'])
    video_path = os.path.join(experiment_path, protocol['videos'][args.video])

    # read_in blinky data
    blinky_fn = os.path.join(experiment_path,
            'processed/{}_blinky_signal.wav'.format(args.video))
    _, blinky_sig = wavfile.read(blinky_fn)

    # valid blinky mask
    blinky_valid_mask = np.ones(blinky_sig.shape[1], dtype=np.bool)
    blinky_valid_mask[protocol['blinky_ignore']] = False

    # read in the groundtruth locations
    source_loc_fn = os.path.join(experiment_path,
            'processed/{}_source_locations.json.gz'.format(args.video))
    if os.path.exists(source_loc_fn):
        source_locations = jsongzip.load(source_loc_fn)
    else:
        source_locations = None

    # Get the Neural Network Model
    with open(args.config, 'r') as f:
        config = json.load(f)

    # import model and use MSE
    nn = models[config['model']['name']](*config['model']['args'], **config['model']['kwargs'])
    chainer.serializers.load_npz(config['model']['file'], nn)

    # Create some random colors (BGR)
    color_gt = [34, 139, 34]
    color_est = [51, 51, 255]

    # find start of segment
    if args.video in protocol['video_segmentation']:
        f_start, f_end = protocol['video_segmentation'][args.video][args.path]
        f_start = int(f_start * protocol['video_info']['fps'])
        f_end = int(f_end * protocol['video_info']['fps'])
        video_output_name = '{}_{}.{}'.format(args.video, args.path, args.format)
    else:
        f_start, f_end = 0, None
        video_output_name = '{}.{}'.format(args.video, args.format)

    i_frame = f_start
    nf = 0
    thresh = 0.4 if args.video in ['speech', 'noise'] else 0.25

    if source_locations is not None:
        while source_locations[0][0] < i_frame:
            source_locations.pop(0)

    with ThreadedVideoStream(video_path, start=f_start, end=f_end) as cap:

        cap.start()

        if args.save is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            shape = cap.get_shape()
            writer = cv2.VideoWriter(TEMP_VIDEO_FILE, fourcc, cap.get_fps(), shape, isColor=True)

        while cap.is_streaming():

            frame = cap.read()

            if frame is None:
                break

            if i_frame >= nf:
                # perform localization
                block = blinky_sig[i_frame-nf:i_frame+nf+1,blinky_valid_mask]
                nn_in = np.mean(block, axis=0)
                nn_in = nn_in[None,:].astype(np.float32)

                if block.max() > thresh:
                    y, x = nn(nn_in).data[0,:]
                    frame = cv2.circle(frame, (x,y), 5, color_est, -1)

                # check if groundtruth is available for that location
                if source_locations is not None and i_frame == source_locations[0][0]:
                    _, [y, x] = source_locations.pop(0)
                    frame = cv2.circle(frame, (x,y), 5, color_gt, -1)

                if not args.no_show:
                    cv2.imshow('frame', frame)

                if args.save is not None:
                    writer.write(frame)

            i_frame += 1

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        if args.save is not None:
            writer.release()

            # now copy the sound
            import subprocess, datetime

            if not os.path.exists(args.save):
                os.mkdir(args.save)

            if args.path is not None:
                output_name = os.path.join(args.save, video_output_name)

            start_time_offset = datetime.timedelta(seconds=f_start / protocol['video_info']['fps'])
            duration = datetime.timedelta(seconds=(f_end - f_start) / protocol['video_info']['fps'])
            print(start_time_offset, duration)
            cmd = [
                    'ffmpeg',
                    '-y',  # overwrite output file if existing
                    '-ss', str(start_time_offset), '-t', str(duration), '-i', video_path,
                    '-i', TEMP_VIDEO_FILE,
                    '-c', 'h264',
                    '-acodec', 'mp3',
                    '-f', args.format,
                    '-map', '0:a:0',
                    '-map', '1:v:0',
                    '-shortest',
                    os.path.join(args.save, video_output_name),
                    ]
            print(' '.join(cmd))
            subprocess.call(cmd)
            os.remove(TEMP_VIDEO_FILE)

    cv2.destroyAllWindows()

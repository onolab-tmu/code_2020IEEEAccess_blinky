import argparse, os, json
import numpy as np
from scipy.io import wavfile
import jsongzip

# Use the parallel sweeps for training and the diagonal sweeps for testing
test_sweeps = ['parallel_short', 'parallel_long']
train_sweeps = ['diagonal']

def in_interval(n, interval):
    return interval[0] <= n and n < interval[1]

if __name__ == '__main__':

    video_choices = ['noise', 'speech', 'hori_1', 'hori_2', 'hori_3', 'hori_4', 'hori_5',]

    parser = argparse.ArgumentParser(description='Extract locations of blinkies and moving source from vide')
    parser.add_argument('protocol', type=str,
            help='The protocol file containing the experiment metadata')
    parser.add_argument('-n', '--num_frames', type=int, default=5,
            help='The number of frames to average to create one input vector')
    parser.add_argument('-v', '--validation_frac', type=int, default=10,
            help='The number of examples out of one which is kept for validation')
    args = parser.parse_args()

    # get the path to the experiment files
    experiment_path = os.path.split(args.protocol)[0]

    # Read some parameters from the protocol
    with open(args.protocol,'r') as f:
        protocol = json.load(f)

    # The end result is two arrays
    data = dict(
        test=[],
        validation=[],
        train=[],
        hori_test=[],
        )

    not_test_counter = 0
    nf = args.num_frames // 2  # the number of frames on each side to average

    thresholds = protocol['detection_thresholds']

    for video in video_choices:

        # read_in blinky data
        blinky_fn = os.path.join(experiment_path,
                'processed/{}_blinky_signal.wav'.format(video))
        _, blinky_sig = wavfile.read(blinky_fn)

        # read in the groundtruth locations
        source_loc_fn = os.path.join(experiment_path,
                'processed/{}_source_locations.json.gz'.format(video))
        if os.path.exists(source_loc_fn):
            source_locations = jsongzip.load(source_loc_fn)
        elif video.startswith('hori'):
            # for horiike video, same location for all frames
            loc = protocol['hori_loc'][video]
            n_frames = blinky_sig.shape[0]
            source_locations = list(zip(range(n_frames), [loc] * n_frames))

        # list of frames to ignore
        if video in protocol['mask_ignore_frames']:
            ignore_list = set(protocol['mask_ignore_frames'][video])
        else:
            ignore_list = set([])  # nothing

        # valid blinky mask
        blinky_valid_mask = np.ones(blinky_sig.shape[1], dtype=np.bool)
        blinky_valid_mask[protocol['blinky_ignore']] = False

        # the intervals
        if video in protocol['video_segmentation']:
            parallel_short = protocol['video_segmentation'][video]['parallel_short']
            parallel_long = protocol['video_segmentation'][video]['parallel_long']
            diagonal = protocol['video_segmentation'][video]['diagonal']

        for (frame, src_loc) in source_locations:

            # skip frame in the ignore list
            if len(ignore_list.intersection(set(range(frame-nf, frame+nf+1)))) > 0:
                continue

            frames = blinky_sig[frame-nf:frame+nf+1,blinky_valid_mask]
            if frames.shape[0] < args.num_frames:
                continue

            tau = frame / protocol['video_info']['fps']
            block = blinky_sig[frame-nf:frame+nf+1,blinky_valid_mask]

            # skip frames that are not very active
            if np.max(block) < thresholds[video]:
                continue

            in_vec = np.mean(block, axis=0)

            example = [in_vec.tolist(), src_loc]

            if video.startswith('hori'):
                data['hori_test'].append(example)

            elif in_interval(tau, parallel_short) or in_interval(tau, parallel_long):

                not_test_counter += 1

                if not_test_counter % args.validation_frac == 0:
                    data['validation'].append(example)

                else:
                    data['train'].append(example)

            elif in_interval(tau, diagonal):

                data['test'].append(example)

    # save the data
    dest_dir = os.path.join(experiment_path, 'learning_data')
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    data_fn = 'data'

    data_fn += '.json.gz'
    jsongzip.dump(os.path.join(dest_dir, data_fn), data)

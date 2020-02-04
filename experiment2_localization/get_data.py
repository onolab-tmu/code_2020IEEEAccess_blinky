# Copyright (c) 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This script can be used to download the data used in the experiments.
"""
import argparse
import json
import os
import hashlib
from urllib.request import urlretrieve, urlopen


def md5(fname, CHUNK=16384):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def url_retrieve_md5(url, filename, CHUNK=16384):
    """
    Retrieve URL and compute MD5 at the same time
    """

    response = urlopen(url)
    hash_md5 = hashlib.md5()

    with open(filename, "wb") as f:
        for chunk in iter(lambda: response.read(CHUNK), b""):
            f.write(chunk)
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def recursive_mkdir(path):
    """
    Utility to create a directory with all necessary subdirectories.
    """

    if path == "":
        return
    elif os.path.exists(path):
        assert os.path.isdir(path), f"File {path} exists but is not a directory"
        return

    base, top = os.path.split(path)
    recursive_mkdir(base)
    os.mkdir(path)
    return


def get_data():

    parser = argparse.ArgumentParser(description="Download all the videos from Zenodo")
    parser.add_argument(
        "protocol_file",
        type=str,
        help="The protocol file containing the video information",
    )
    args = parser.parse_args()

    protocol_filename = args.protocol_file
    data_folder = os.path.split(protocol_filename)[0]

    # Get the list of videos
    with open(protocol_filename, "r") as f:
        protocol = json.load(f)
        videos = protocol["videos"]

    # zenodo base URL
    base_url = protocol["zenodo_url"]

    # Download and check all the videos
    for name, path in videos.items():
        video_folder, video_filename = os.path.split(path)

        # get the expected MD5 checksum
        expected = protocol["videos_md5"][name]

        url = os.path.join(base_url, video_filename)
        filename = os.path.join(data_folder, path)

        if os.path.exists(filename):
            print(f"File {path} already exists.")
            checksum = md5(filename)

        else:
            # make sure the folder exists
            recursive_mkdir(os.path.join(data_folder, video_folder))

            print(f"Downloading {url}... ", end="", flush=True)

            # download the video
            checksum = url_retrieve_md5(url, filename)

            print("done.")

        if checksum != expected:
            print(
                f"Checksum error for {filename}. Expected {expected}, "
                f"got {checksum} instead."
            )


if __name__ == "__main__":
    get_data()

import argparse
import os
import requests

"""
Sends all images in a folder to the autograder server.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--netid", required=True, help="student's NetID")
parser.add_argument("--token", required=True, help="student's token")
parser.add_argument("--image-dir", required=True, help="submission directory of 128x128 images")
parser.add_argument("--server", required=True, help="IP address of grading server")
args = parser.parse_args()

images = [x for x in os.listdir(args.image_dir)]
files = {}
for image in images:
    with open(os.path.join(args.image_dir, image), "rb") as bin_data:
        files[image] = bin_data.read()

payload = {"netid": args.netid, "token": args.token}
res = requests.post(args.server, files=files, data=payload)
print(res.text)

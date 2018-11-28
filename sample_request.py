import argparse
import datetime
import os

import requests

"""
Sends all images in a folder to the auto grader server.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--netid', required=True, help='student\'s NetID')
parser.add_argument('--token', required=True, help='student\'s token')
parser.add_argument('--image-dir', required=True,
                    help='submission directory of 128x128 images')
parser.add_argument('--server', required=True,
                    help='IP address of grading server')
parser.add_argument('--history-csv', required=True,
                    help='the place to save the history')
parser.add_argument('--comment', required=True,
                    help='comment about this submission')
args = parser.parse_args()

images = [x for x in os.listdir(args.image_dir)]
files = {}
for image in images:
    with open(os.path.join(args.image_dir, image), 'rb') as bin_data:
        files[image] = bin_data.read()

payload = {"netid": args.netid, "token": args.token}
res = requests.post(args.server, files=files, data=payload)
print(res.text)
with open(args.history_csv, 'r+') as h:
    history_lines = h.readlines()
    if len(history_lines) == 0:
        h.write('Time, Result, Comment\n')
    now = datetime.datetime.now()
    h.write('%s, %s, %s\n' % (
        now.strftime("%Y-%m-%d %H:%M"), res.text, args.comment))

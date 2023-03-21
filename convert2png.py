import cv2
import os

# read form full vids
name = 'beam_video_fast_noisy_128'
vidcap = cv2.VideoCapture('./full_vid/{:s}.mp4'.format(name))
success,image = vidcap.read()
count = 0

if not os.path.exists("./vids/{:s}".format(name)):
  os.mkdir("./vids/{:s}".format(name))

while success:

  cv2.imwrite("./vids/{:s}/{:06d}.png".format(name,count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  # print('Read a new frame: ', success)
  count += 1

print("Total frames: ", count)
import cv2
import numpy as np
import glob

# read form full vids
name = 'gun'
frame_rate = 30

imgs_dir = './output/{:s}'.format(name)
video_dir = './full_vid/'

img_array = []
# files are in order! 
for filename in sorted(glob.glob('{:s}/*.png'.format(imgs_dir))):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
print(size)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_dir+'mm_{:s}.mp4'.format(name), fourcc, frame_rate, (size[0], size[1]))

for i in range(len(img_array)):
    video_writer.write(img_array[i])

video_writer.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# read form full vids
name = 'baby'
video_dir = './full_vid/'
frame_rate = 30
vidcap = cv2.VideoCapture('./full_vid/mm_{:s}.avi'.format(name))
vidcapF = cv2.VideoCapture('./full_vid/mm_{:s}_filt.avi'.format(name))

success,image = vidcap.read()
successF,imageF = vidcapF.read()

size = (image.shape[0], image.shape[1])

imgs = []
count = 0
while success:

    result_img = np.abs(image - imageF)
    print((image == imageF).any())
    # cv2.imwrite("./vids/{:s}/{:06d}.png".format(name,count), image)     # save frame as JPEG file      
    imgs.append(result_img)
    success,image = vidcap.read()
    successF,imageF = vidcapF.read()
    # print('Read a new frame: ', success)
    count += 1

print("Total frames: ", count)

out = cv2.VideoWriter(video_dir+'diff_{:s}.avi'.format(name),cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
# out = cv2.VideoWriter(video_dir+'mm_{:s}.mp4'.format(name),cv2.VideoWriter_fourcc(*'X264'), 15, size)

for i in range(len(imgs)):
    out.write(imgs[i])
out.release()
import cv2

# read form full vids
name = 'machines'
vidcap = cv2.VideoCapture('./full_vid/{:s}.mp4'.format(name))
success,image = vidcap.read()
frame_rate = 24

height, width, layers = image.shape
size = (width,height)
x1 = 200
x2 = width
y1 = 200
y2 = height

frame_start = 1218
frame_end = 1350

frame = 0
imgs = []
while success:
  if frame > frame_start and frame < frame_end:
    #image = image[x1:x2, y1:y2,:]
    imgs.append(image)

  success,image = vidcap.read()
  frame += 1

out = cv2.VideoWriter('./full_vid/{:s}_crop.avi'.format(name),cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
for i in range(len(imgs)):
    out.write(imgs[i])
out.release()

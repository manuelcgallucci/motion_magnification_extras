import cv2
import numpy as np
import glob
import os 
import matplotlib.pyplot as plt

points = []
type_ = None

def define_type(definition):
    global type_
    type_ = definition

def draw_cross(img, x, y, color, thickness=2, offset=5):
    cv2.line(img, (x+offset,y+offset), (x-offset,y-offset), color, thickness)
    cv2.line(img, (x+offset,y-offset), (x-offset,y+offset), color, thickness)

def mouse_callback(event, x, y, flags, image):
    if event == cv2.EVENT_LBUTTONDOWN:
        colors_bgr = [
            [0, 0, 255], 
            [0, 255, 0],   
            [255, 0, 0],  
        ]
        if type_ == "pixel":
            draw_cross(image, x, y, colors_bgr[len(points) // 2])
        elif type_ == "line" and len(points) == 0:
            draw_cross(image, x, y, colors_bgr[len(points) // 2])

        cv2.imshow("Choose pixels", image)
        points.append((x,y))

# These 3 functions could be merged into one but oh well, ...
def save_channel_plot(seq, show=True, path=None, format="bgr"):
    if format != "bgr":
        print("Plotting rgb channels only supported using BGR format!")

    plt.figure()
    plt.plot(seq[:,2], "r")
    plt.plot(seq[:,1], "g")
    plt.plot(seq[:,0], "b")
    plt.legend(["Red channel", "Green channel", "Blue channel"])
    plt.title("Channels")
    if path is not None: plt.savefig(path)
    if show: plt.show()
    plt.close()

def save_plot(seq, title, show=True, path=None, xaxis=None):
    plt.figure()
    if xaxis is None:
        plt.plot(seq)
    else:
        plt.plot(xaxis, seq)
    plt.title(title)
    if path is not None: plt.savefig(path)
    if show: plt.show()
    plt.close()

def save_img(img, title, path, format="bgr"):
    if format == "bgr":
        img = img[:,:,::-1]
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def main(name = 'beam_video_fast_modes', fps=30.0):
    imgs_dir = './output/{:s}'.format(name)
    
    #vids_dir = './vids/{:s}'.format(name)
    #imgs_dir = vids_dir

    first_image = True
    sequence = []

    # files are in order!
    for filename in glob.glob('{:s}/*.png'.format(imgs_dir)):
        
        img = cv2.imread(filename)
        
        if first_image and len(points) == 0:
            cv2.imshow("Choose pixels", img)
            
            print("Choose the mode 1 or 2")
            while type_ is None:
                key = cv2.waitKey(1)
                if key == ord("1"):
                    print("Mode: One pixel")
                    define_type("pixel")
                elif key == ord("2"):
                    print("Mode: Line")
                    define_type("line")
            
            cv2.setMouseCallback("Choose pixels", mouse_callback, img)
            for i in range(len(points)):
                points.pop()

            if type_ == "pixel":
                while True:
                    cv2.waitKey(1)
                    if len(points) > 0:
                        break
            elif type_ == "line":
                while True:
                    cv2.waitKey(1)
                    if len(points) > 1:
                        break            
        
        if type_ == "pixel": 
            x, y = points[0]
            #cv2.circle(img, (x,y), 3, (255,255,255), -1)
            #cv2.imshow("Choose pixels", img)
            sequence.append(img[y, x,:])
            if first_image:
                print((x,y))
        elif type_ == "line":
            x1, y1 = points[0]
            x2, y2 = points[1]
            # Line has to be horizontal or vertical 
            if (x1-x2)**2 < (y1-y2)**2:
                x2 = x1
                aux = y1
                y1 = min(y1,y2)
                y2 = max(aux,y2)
                line_type = "vertical"
            else:
                y2 = y1
                aux = x1
                x1 = min(x1,x2)
                x2 = max(aux,x2)
                line_type = "horizontal"
            
            if first_image:          
                print((x1, y1), (x2, y2))      
                draw_cross(img, x2,y2, (0,0,255))
                cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.imshow("Choose pixels", img)
                cv2.waitKey(5) # eq. to sleep

            #print(line_type)
            if line_type == "vertical":
                sequence.append(img[y1:y2, x1,:])
            elif line_type == "horizontal":
                sequence.append(img[y1, x1:x2,:])
        
        if first_image:
            cv2.imwrite("./chosen_pixels.png", img) 
            cv2.destroyAllWindows()
            
        first_image = False

    # Colors are in BGR space 
    # Computation of lightness for RGB colors NORMALIZED!
    # gamma = 2.2
    # Y = .2126 * R^gamma + .7152 * G^gamma + .0722 * B^gamma
    # L* = 116 * Y ^ 1/3 - 16
    if type_ == "pixel":
        sequence = np.array(sequence) / 255.0
        
        gamma = 2.2
        Y =  .2126 * np.power(sequence[:,2], gamma) + .7152 * np.power(sequence[:,1], gamma) + .0722 * np.power(sequence[:,0],gamma)
        lightness = 116 * np.power(Y, 1/3) - 16

        light_fft = np.fft.fft(lightness)
        light_fft = np.fft.fftshift(light_fft)
        light_fft = np.abs(light_fft)


        save_plot(lightness, "Lightness", path="./pixel_lightness.png", show=False)
        xaxis=[i*(1/ fps) for i in range(len(light_fft)//2 )]
        save_plot(light_fft[len(light_fft)//2+1:], "Lightness FFT", path="./pixel_lightness_fft.png", show=False, xaxis=xaxis)

        save_channel_plot(sequence, path="./pixel_channels.png", show=False)

        # save_plot(sequence[:,2], "Red channel", path="./pixel_r.png", show=False)
        # save_plot(sequence[:,1], "Green channel", path="./pixel_g.png", show=False)
        # save_plot(sequence[:,0], "Blue channel", path="./pixel_b.png", show=False)

    elif type_ == "line": # Here sequence is a list of arrays ( length_line x 3)
        if line_type == "horizontal":
            save_img( np.array(sequence), "seq evolution", "./seq.png")
        elif line_type == "vertical":
            for s in range(len(sequence)):
                sequence[s] = np.flip(sequence[s])
            print(np.array(sequence).shape)
            sequence = np.rot90(np.array(sequence))[:,1:,:]

            sequence = sequence[:,300:,:]
            save_img(sequence, "seq evolution", "./seq.png", format="rgb")
            print(sequence.shape)
            max_seq = []
            for t in range(sequence.shape[1]):
                max_seq.append(np.argmax(np.mean(sequence[:,t,:], axis=1)))
            
            #max_seq = max_seq[:299]
            max_seq = np.array(max_seq)
            save_plot(max_seq, "max_seq", path="./max_seq.png", show=False)
            max_seq_fft = np.fft.fft(max_seq- np.mean(max_seq))
            max_seq_fft = np.fft.fftshift(max_seq_fft)
            max_seq_fft = np.abs(max_seq_fft)

            xaxis=[i*(1/ fps) for i in range((len(max_seq_fft)-1)//2)]
            # print(tuple(max_seq_fft[len(max_seq_fft)//2+1:]))
            save_plot(max_seq_fft[len(max_seq_fft)//2+1:], "Max seq", path="./max_seq_fft.png", show=False, xaxis=xaxis)
            
    return 1

if __name__ == "__main__":
    main()

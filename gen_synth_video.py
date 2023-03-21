import numpy as np
import cv2
    

out_name = 'beam_video_fast_noisy_128'
noisy = 128

# Run the simulation
fps = 30
tmax = 10.0
t = 0.0

# size = (480, 640, 3)
size = (200, 300, 3)

# Initialize the OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('{:s}.mp4'.format(out_name), fourcc, fps, (size[1], size[0]))

# Create sinusoidal modes
L = 180

n_modes = 3
f_mode = 7.0
freqs = [f_mode*(i+1) for i in range(n_modes)]
wavelengths = [L//(i+1) for i in range(n_modes)]
amplitudes =  [1, 0, 0] # [2,2,2] 

# freqs = [5, 10, 15]  # in Hz
# phases = [0, np.pi/2, np.pi]
# wavelengths = [L / f for f in freqs]
# amplitudes = [20, 10, 5]

while t < tmax:
    # Update the pendulum state
    # Draw the pendulum on a black background
    if noisy == 0:
        img = np.zeros(size, dtype=np.uint8)
    else:
        img = np.random.randint(0, noisy, size=size,dtype=np.uint8)

    # Generate frame
    disp = np.zeros((L, 1))
    x = np.linspace(0, L, L)
    for i, f in enumerate(freqs):
        wavelength = wavelengths[i]
        amplitude = amplitudes[i]

        # y = 2 * amplitude * np.sin(2*np.pi*f*t/fps + phase + (1/L) *x)[:,None]
        y = 2 * amplitude * np.cos(2*np.pi*f*t + np.pi/2) * np.sin(x*np.pi*(i+1) / L)[:,None]
        disp += y
        # y = np.clip(y, 0, 255).astype(np.uint8)
    
    for position, pixel_displacement in enumerate(disp):
        
        # img[int(size[0] // 2 + pixel_displacement)-3,size[1] // 2 - L//2 + position,:] = 128
        # img[int(size[0] // 2 + pixel_displacement)-2,size[1] // 2 - L//2 + position,:] = 128
        img[int(np.round(size[0] // 2 + pixel_displacement)-3),size[1] // 2 - L//2 + position,:] = 32
        img[int(np.round(size[0] // 2 + pixel_displacement)-2),size[1] // 2 - L//2 + position,:] = 64
        img[int(np.round(size[0] // 2 + pixel_displacement)-1),size[1] // 2 - L//2 + position,:] = 128
        img[int(np.round(size[0] // 2 + pixel_displacement)),size[1] // 2 - L//2 + position,:] = 255
        img[int(np.round(size[0] // 2 + pixel_displacement)+1),size[1] // 2 - L//2 + position,:] = 128
        img[int(np.round(size[0] // 2 + pixel_displacement)+2),size[1] // 2 - L//2 + position,:] = 64
        img[int(np.round(size[0] // 2 + pixel_displacement)+3),size[1] // 2 - L//2 + position,:] = 32

        # img[int(size[0] // 2 + pixel_displacement)+2,size[1] // 2 - L//2 + position,:] = 128
        # img[int(size[0] // 2 + pixel_displacement)+3,size[1] // 2 - L//2 + position,:] = 128
    # img[size[0] // 2,size[1] // 2 - L//2:size[1] // 2 + L//2,:] = 255
    
    # Write the frame to the video file and display it
    video_writer.write(img)
    
    # Increment the simulation time
    t += 1/fps

# Release the video writer and destroy the OpenCV window
video_writer.release()
cv2.destroyAllWindows()
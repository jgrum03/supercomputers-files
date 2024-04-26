import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm # Color maps
from projFuncs import *
from processImg import *
import PIL
from PIL import Image as Im
from tqdm import tqdm

# Make video
import cv2
import os
from natsort import natsorted

# Parallelize
from joblib import Parallel, delayed

### Using geometrized units; c=G=1.

pixel_lengthi = 1 # pixel length for the initial image
pixel_lengthf = 1 # pixel length for the final image
x1 = -20 # location of where we observe the final image
x2 = 50 # location of the initial image. Keep it a positive number and the final image at a negative location for later parts of this program to work.
# make SURE that |x2| > |x1| so that using the bounds argument in the integrating function works.
y_bound = 2000
z_bound = 2000

# Video parameters
vidFolder = "video"
fps = 8
video_name = 'figures/lensing.mp4'

# set up images
imageName = "deepfield.png"
initialImage = imageTakeInner(imageName)
y_sizei, z_sizei, x_sizei = initialImage.shape
y_sizef = 500
z_sizef = 500
x_sizef = x_sizei # needs to be the same to transfer the information between the matrices.
finalImage = np.zeros(np.array([y_sizef, z_sizef, x_sizef]), dtype='int')

# intial image position values
y_positionsi = np.arange(0, y_sizei, 1)
z_positionsi = np.arange(0, z_sizei, 1)
y_positionsi = pixel_lengthi*(y_positionsi - y_sizei/2)
z_positionsi = pixel_lengthi*(z_positionsi - z_sizei/2)

nFrames = 8 # Number of frames
pixelStepFrame = 2 # how many pixels to move each frame. Must be an integer.

plt.style.use('dark_background')

print("Started.")

# for f in tqdm(range(nFrames)):
def process(f):
    y_centeri = pixel_lengthi*y_sizei/2
    z_centeri = pixel_lengthi*z_sizei/2-(nFrames*pixelStepFrame)/2+f*pixelStepFrame+30

    # final image position values
    y_positionsf = np.arange(0, y_sizef, 1)
    z_positionsf = np.arange(0, z_sizef, 1)
    y_positionsf = pixel_lengthf*(y_positionsf - y_sizef/2)
    z_positionsf = pixel_lengthf*(z_positionsf - z_sizef/2)

    # for shift in range(30)
    y_centerf = pixel_lengthf*y_sizef/2
    z_centerf = (pixel_lengthf*z_sizef/2)-(nFrames*pixelStepFrame)/2+f*pixelStepFrame

    for i in tqdm(range(y_sizef)):
        for j in range(z_sizef):
            z = z_positionsf[j]
            y = y_positionsf[i]
            x = x1
            vx = 1
            vy = 0
            vz = 0
            u = integrate_EOM(np.array([x,y,z]), np.array([vx,vy,vz]),1,np.array([x2,y_bound + 1,z_bound + 1]))
            Uu = u[1:,-1,0] # ultimate
            if ((Uu[0] - x2) >= 0):
                k,l = findPixel(y_centeri, z_centeri, x2, pixel_lengthi, Uu[:3], Uu[3:])
                if ((k > -1) and (k < y_sizei) and (l > -1) and (l < z_sizei)):
                    finalImage[i,j,:] = initialImage[k,l,:]

    plt.style.use('dark_background')
    fig,ax = plt.subplots(1)
    ax = plt.gca()
    ax.imshow(finalImage)
    ax.add_patch(plt.Circle((z_sizef/2,y_sizef/2), Rs, color='red'))
    ax.add_patch(plt.Circle((z_sizef/2,y_sizef/2), 1.5*Rs, color='red', fill=False, linewidth=2.0, linestyle=':'))
    ax.add_patch(plt.Circle((z_sizef/2,y_sizef/2), 2.598*Rs, color='red', fill=False, linewidth=2.0, linestyle='--'))
    plt.savefig(str(vidFolder)+"/Lensed_"+str(f)+"_"+imageName)
    plt.close(fig)

Parallel(n_jobs=1)(delayed(process)(k) for k in tqdm(range(nFrames)))

print("Frames Done.")
print("Making Video...")

# Make video from the images

images = [img for img in os.listdir(vidFolder) if img.endswith(".png")]
images = natsorted(images) # Sort the frames in the correct order
frame = cv2.imread(os.path.join(vidFolder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, fps, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(vidFolder, image)))

cv2.destroyAllWindows()
video.release()


print("Finished.")

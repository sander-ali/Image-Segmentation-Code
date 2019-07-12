#Demo for image segmentation using various methods
#With Scikit-image package
#For documentation please refer to
# https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
# importing the packages
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

#Function to show an image
def show_image(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

#Import the image from your directory
image = io.imread('girl.jpg') 
plt.imshow(image);

#Converting the RGB image into grayscale
image_gray = color.rgb2gray(image) 
show_image(image_gray);

#Segmentation using Active Contour
#Input arguments center, radius, resolution
def find_circles(cent, rad, points ):
    """ Generate points to define a circle """
    radians = np.linspace(0, 2*np.pi, points)
    # transforming the cartision co-ordinates to polar co-ordinates
    c = cent[1] + rad*np.cos(radians)
    r = cent[0] + rad*np.sin(radians)
    
    return np.array([c, r]).T
# Exclude last point because a closed path should not have duplicate points
#since we have given 200 points it will generate a circle with the given number
#of points
points = find_circles([150, 250], 110, 800)[:-1]

fig, ax = show_image(image)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)

#Applying segmentation using active snakes
snake = seg.active_contour(image_gray, points)
fig, ax = show_image(image)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

#Optimizing the segmentation using controllable paramaters alpha and beta
snake = seg.active_contour(image_gray, points,alpha=0.95,beta=0.8)
fig, ax = show_image(image)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

#Segmentation using Random Walker method
"""The random walker algorithm expects a label image as input.
So we will have the bigger circle that encompasses the person’s entire
and another smaller circle near the middle of the face. """
#Reference paper "Random Walks for Image Segmentation" IEEE TPAMI 2006
image_labels = np.zeros(image_gray.shape, dtype=np.uint8)
indices = draw.circle_perimeter(150,250,80)
image_labels[indices] = 1
image_labels[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2
show_image(image_labels);
#using random walker method 
image_segmented = seg.random_walker(image_gray, image_labels)
# Check out the results
fig, ax = show_image(image_gray)
ax.imshow(image_segmented == 1, alpha=0.3);

#similar to active snakes lets optimize the random walker method
image_segmented = seg.random_walker(image_gray, image_labels, beta = 6000)
# Check our results
fig, ax = show_image(image_gray)
ax.imshow(image_segmented == 1, alpha=0.3);

#Now lets try with unsupervised methods
#SLIC (Simple Linear Iterative Clustering)
#Reference paper "SLIC Superpixels Compared to State-of-the-Art Superpixel 
#Methods" IEEE TPAMI 2012
image_slic = seg.slic(image,n_segments=25)
#The segment 25 means the number of segmented classes
# label2rgb replaces each discrete label with the average interior color
show_image(color.label2rgb(image_slic, image, kind='avg'));

#using Felzenswalb method (minimum spanning tree clustering)
#Reference paper "Efficient Graph-Based Image Segmentation" IJCV 2004

image_felzenszwalb = seg.felzenszwalb(image) 
show_image(image_felzenszwalb);
#Lot of regions right!! lets calculate the number of unique regions
np.unique(image_felzenszwalb).size

#now lets recolor each of the region using SLIC method
image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, image, kind='avg')
show_image(image_felzenszwalb_colored);

#You can use scale parameters to use lower number of regions.
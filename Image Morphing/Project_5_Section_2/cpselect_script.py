
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os, sys

'''
	SCRIPT FOR CIS 581 - PROJECT 5

	Usage:
		To run the script, execute:
			python cpselect_script.py [source_image_filename] [target_image_filename]

		This script will allow you to point and click to set correspondences between
		two input images. You will set exactly 32 correspondences between the images, for
		a total of 64 points. For consistency, it's recommended you click the correspondences 
		in the order defined below, named "vertices". You must alternate between clicking 
		the same feature in the source and target images. For example:

			1. First, click the left pupil of the source image
			2. Then, click the left pupil of the target image
			3. Click the right pupil of the source image
			4. Click the right pupil of the target image
			5. Click the corner of the left eye in the source image
			...
		
		Continue the process until all 64 points are set.

		After you're done, the correspondences will automatically be saved as numpy files
		(.npy), which can be uploaded to your Colab notebook.
'''

NUM_POINTS = 32

vertices = [
  (0, 'left pupil'),
  (1, 'right pupil'),
  (2, 'inner corner of left eye'),
  (3, 'outer corner of left eye'),
  (4, 'inner corner of right eye'),
  (5, 'outer corner of right eye'),
  (6, 'inner endpoint of left eyebrow'),
  (7, 'inner endpoint of right eyebrow'),
  (8, 'outer endpoint of left eyebrow'),
  (9, 'outer endpoint of right eyebrow'),
  (10, 'midpoint of right eyebrow'),
  (11, 'midpoint of left eyebrow'),
  (12, 'left nostril'),
  (13, 'right nostril'),
  (14, 'left side of the nose'),
  (15, 'right side of the nose'),
  (16, 'top of the lip, under the left nostril'),
  (17, 'midpoint of the top of the lip'),
  (18, 'top of the lip, under the right nostril'),
  (19, 'right corner of the mouth'),
  (20, 'left corner of the mouth'),
  (21, 'bottom point of the bottom lip'),
  (22, 'left side of the chin'),
  (23, 'right side of the chin'),
  (24, 'top-left corner of forehead'),
  (25, 'top-right corner of forehead'),
  (26, 'left side of the jawline'),
  (27, 'right side of the jawline'),
  (28, 'top-left corner of the image'),
  (29, 'top-right corner of the image'),
  (30, 'bottom-left corner of the image'),
  (31, 'bottom-right corner of the image'),
]


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('Please provide two images.')
		sys.exit(1)

	# Setting up filenames
	img1_fname = sys.argv[1]
	img2_fname = sys.argv[2]
	
	pts1_fname = img1_fname[:img1_fname.rfind('.')] + '_points'
	pts2_fname = img2_fname[:img2_fname.rfind('.')] + '_points'
	print('Saving to:', pts1_fname, ",", pts2_fname)

	# Display images
	img1 = np.array(Image.open(img1_fname).convert('RGB'))
	img2 = np.array(Image.open(img2_fname).convert('RGB'))

	fig, (Ax0, Ax1) = plt.subplots(1, 2, figsize = (20, 20))
	Ax0.imshow(img1)
	Ax0.axis('off')
	Ax1.imshow(img2)
	Ax1.axis('off')

	# Use ginput to get the correspondences
	print("Running ginput:")
	x = plt.ginput(2 * NUM_POINTS, timeout=-1)
	#print(x)
	plt.close()

	img1_pts = np.array(x[0::2])
	img2_pts = np.array(x[1::2])

	# Save our arrays
	print('Saving points...')
	np.save(pts1_fname, img1_pts)
	np.save(pts2_fname, img2_pts)
	# print(img1_pts)
	# print(img2_pts)
	print('Done!')

	# Show results
	plt.title('Image 1 points:')
	plt.imshow(img1)
	plt.scatter(img1_pts[:,0], img1_pts[:,1])
	plt.show()

	plt.title('Image 2 points:')
	plt.imshow(img2)
	plt.scatter(img2_pts[:,0], img2_pts[:,1])
	plt.show()



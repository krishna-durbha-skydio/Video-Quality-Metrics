"""
NIQE
"""
# Importing Libraries
import numpy as np
import scipy
import cv2
from numba import jit

import functions.NSS_tools as NSS_tools


class NIQE:
	def __init__(self,
		block_size:tuple,
		mu_pristine_param:np.array,
		covariance_pristine_param:np.array,
		C:float=1e-3
	):
		"""
		Parameters for NIQE
		Args:
			block_size (tuple): Block-size
			scale_factor
		"""
		self.block_size = block_size
		self.mu_pristine_param = mu_pristine_param
		self.covariance_pristine_param = covariance_pristine_param
		self.C = C


	def compute_features(self,
		img:np.asarray,
		blocksizerow:int,
		blocksizecol:int
	):
		"""
		Computing NIQE features
		Args:
			img (np.array): Image
			blocksizerow (int): Height of block
			blocksizecol (int): Width of block
		"""
		# MSCN
		mscn, _, _ = NSS_tools.compute_image_mscn_transform(img, C=self.C, extend_mode='nearest')
		mscn = mscn.astype(np.float32)

		# Extracting on Patches
		features = NSS_tools.extract_on_patches(mscn, blocksizerow, blocksizecol)

		return features


	def compute_score(self,
		img:np.array,
		scale_factor:float
	):
		# Dimensions of Image
		h, w = img.shape

		# Block Dimensions
		blocksizerow, blocksizecol = self.block_size

		if (h < blocksizerow) or (w < blocksizecol):
			assert False, "Input frame is too small"
			
		# Ensure that the Patch divides evenly into img
		hoffset = (h % blocksizerow)
		woffset = (w % blocksizecol)

		# Offsetting Image
		if hoffset > 0: 
			img = img[:-hoffset, :]
		if woffset > 0:
			img = img[:, :-woffset]

		img = img.astype(np.float32)

		# Features on Original Scale
		features_scale1 = self.compute_features(img=img, blocksizerow=self.block_size[0], blocksizecol=self.block_size[1])

		# Downsampled Image
		if scale_factor is not None:
			width = int(img.shape[1] * scale_factor)
			height = int(img.shape[0] * scale_factor)

			# Downsampling Image
			downsampled_img = cv2.resize(img, (height,width), interpolation=cv2.INTER_CUBIC)

			# Features on downamsampled scale
			features_scale2 = self.compute_features(img=downsampled_img, blocksizerow=self.block_size[0]//2, blocksizecol=self.block_size[1]//2)

			# Concatenating
			features = np.hstack((features_scale1, features_scale2))
		else:
			features = features_scale1

			# Selecting only original scale NIQE parameters
			self.mu_pristine_param = self.mu_pristine_param[:, :18]
			self.covariance_pristine_param = self.covariance_pristine_param[:18, :18]


		# Distorted Image Paramters
		mu_distorted_param = np.mean(features, axis=0)
		covariance_distorted_param = np.cov(features.T)
		invcov_param = np.linalg.pinv((self.covariance_pristine_param + covariance_distorted_param)/2)

		# Quality Score
		xd = self.mu_pristine_param - mu_distorted_param 
		quality = np.sqrt(np.dot(np.dot(xd, invcov_param), xd.T))[0][0]

		return np.hstack((mu_distorted_param, [quality]))
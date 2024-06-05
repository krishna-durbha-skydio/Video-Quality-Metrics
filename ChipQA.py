"""
Functions to extract ChipQA features
"""
# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)
import scipy
import cv2
from numba import jit

import functions.NSS_tools as NSS_tools
import NIQE


class Compute_ChipQA_features():
	def __init__(self,
		space_time_length:int=5,
		step:int=5,
		C:float=1
	):
		"""
		ChipQA features calculated on a single scale.
		Args:
			space_time_length (int): Length of temporal filter. This will also be used for defining space-time chips. (Default: 5)
			step (int): Frame skip length. Generally equal to space_time_length. (Default: 5)
			C (float): Constant used in MSCN coefficients
		"""
		# Constant
		self.C = C
		if step is not None:
			self.step = step
		else:
			self.step = space_time_length

		# Temporal Filter Kernel: k(t) = t(1 − at)exp(−2at)u(t)
		self.space_time_length = space_time_length
		t = np.arange(0, space_time_length)
		a = 0.5
		self.temporal_filter_kernel = t*(1-a*t)*np.exp(-2*a*t)

		# Flipping it because we apply it on a list containing (V5, V4, V3, V2, V1).
		self.temporal_filter_kernel = np.flip(self.temporal_filter_kernel)

		# Angles
		self.theta = np.arange(0,np.pi,np.pi/6)
		cos_theta = np.cos(self.theta)
		sin_theta = np.sin(self.theta)

		# Space-Time Chips Volumes
		lower_R = int((space_time_length + 1)/2) - 1
		higher_R = int((space_time_length + 1)/2)
		R = np.arange(-lower_R, higher_R)
		R_cos = np.round(np.outer(R, cos_theta))
		R_sin = np.round(np.outer(R, sin_theta))

		# Type Casting
		self.R_cos = R_cos.astype(np.int32)
		self.R_sin = R_sin.astype(np.int32)


		# NIQE
		params = scipy.io.loadmat('parameters/frames_modelparameters.mat')
		self.mu_pristine_param = params['mu_prisparam']
		self.covariance_pristine_param = params['cov_prisparam']


	def compute_BRISQUE_features(self, mscn):
		# Fitting GGD to MSCN
		alpha_m, sigma = NSS_tools.estimate_ggd_features(mscn.copy())

		# Pairwise Product
		pps1, pps2, pps3, pps4 = NSS_tools.paired_product(mscn)

		# Fitting AGGD to pairwise products
		alpha1, N1, bl1, br1, lsq1, rsq1 = NSS_tools.estimate_aggd_features(pps1)
		alpha2, N2, bl2, br2, lsq2, rsq2 = NSS_tools.estimate_aggd_features(pps2)
		alpha3, N3, bl3, br3, lsq3, rsq3 = NSS_tools.estimate_aggd_features(pps3)
		alpha4, N4, bl4, br4, lsq4, rsq4 = NSS_tools.estimate_aggd_features(pps4)

		return np.array([
			alpha_m, sigma,
			alpha1, N1, bl1**2, br1**2,  # (V)
			alpha2, N2, bl2**2, br2**2,  # (H)
			alpha3, N3, bl3**2, br3**2,  # (D1)
			alpha4, N4, bl4**2, br4**2,  # (D2)
		])


	def compute_NIQE_features(self,
		img:np.array,
		block_size:tuple,
		C:float
	):
		"""
		Compute NIQE features on original scale
		"""
		# Dimensions of Image
		h, w = img.shape

		# Block Dimensions
		blocksizerow, blocksizecol = block_size

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
		# MSCN
		mscn, _, _ = NSS_tools.compute_image_mscn_transform(img, C=C, extend_mode='nearest')
		mscn = mscn.astype(np.float32)

		# Extracting on Patches
		features = NSS_tools.extract_on_patches(mscn, blocksizerow, blocksizecol)

		return features
	

	def compute_NIQE_score(self, features):
		# Check no.of scales used for feature extraction
		if features.shape[1] == 18:
			# Only Original Scale
			mu_pristine_param = self.mu_pristine_param[:, :18]
			covariance_pristine_param = self.covariance_pristine_param[:18, :18]
		elif features.shape[1] == 36:
			# Two Levels i.e Original Scale + Downsampled Scale
			mu_pristine_param = self.mu_pristine_param
			covariance_pristine_param = self.covariance_pristine_param
		else:
			assert False, "Unsupported NIQE feature dimensions"

		# Distorted Image Paramters
		mu_distorted_param = np.mean(features, axis=0)
		covariance_distorted_param = np.cov(features.T)
		invcov_param = np.linalg.pinv((self.covariance_pristine_param + covariance_distorted_param)/2)

		# Quality Score
		xd = mu_pristine_param - mu_distorted_param 
		quality = np.sqrt(np.dot(np.dot(xd, invcov_param), xd.T))[0][0]

		return np.hstack((mu_distorted_param, [quality]))


	def extract_features(self,
		video:np.array,
		niqe_block_size
	):
		# Dimensions
		h,w = video.shape[1], video.shape[2]

		# Centers of Space-Time Chips
		center_y, center_x = np.mgrid[self.step:h - self.step*4:self.step*4, self.step:w-self.step*4:self.step*4].reshape(2,-1).astype(int)
		r1 = len(np.arange(self.step, h-self.step*4, self.step*4)) 
		r2 = len(np.arange(self.step, w-self.step*4, self.step*4))

		# Creating Buffers
		img_buffer = np.zeros((self.space_time_length, h, w))
		grad_img_buffer = np.zeros((self.space_time_length, h, w))

		# Creating Outputs
		CGL_features = []
		CGL_features_per_ST_chip = []
		NIQE_features = []
		ST_chip_features = []

		# Iterating over video
		for i in range(0, video.shape[0]):
			# Save Position
			j = i%5

			# Color Conversion
			lab = cv2.cvtColor(src=video[i], code=cv2.COLOR_RGB2LAB)
			yuv = cv2.cvtColor(src=video[i], code=cv2.COLOR_RGB2YUV)
			lab = lab.astype(np.float32)
			
			# Components
			Y = yuv[:,:,0].astype(np.float32)
			grad_Y = NSS_tools.compute_gradient_magnitude(Y)

			# Buffers
			# Luma
			Y_mscn, Y_sigma, _ = NSS_tools.compute_image_mscn_transform(Y)
			Y_sigma_mscn, _, _= NSS_tools.compute_image_mscn_transform(Y_sigma)
			img_buffer[j,:,:] = Y_mscn

			# Luma Gradient
			grad_Y_mscn, _, _ = NSS_tools.compute_image_mscn_transform(grad_Y)
			grad_img_buffer[j,:,:] = grad_Y_mscn
			

			if i > 0:
				# Chroma and Chroma-sigma-map Features
				chroma_features, chroma_sigma_map_features = NSS_tools.extract_chroma_features(lab,C=1)

				# Gradient Features
				grad_features = NSS_tools.second_order_statistical_features(grad_Y_mscn)

				# Luma-sigma-map features
				luma_sigma_map_features = NSS_tools.statistical_features(Y_sigma_mscn)

				# Appending Features
				CGL_features.append(np.concatenate([chroma_features, chroma_sigma_map_features, grad_features, luma_sigma_map_features], axis=0))
			

			# Computing Temporal Features after accumulating frames
			if (i+1)%self.space_time_length == 0 and i > 0:
				# Computing CGL_features_per_ST_chip
				CGL_features_current_ST_chip = CGL_features[-self.space_time_length:]
				CGL_features_per_ST_chip.append(np.std(CGL_features_current_ST_chip, axis=0))

				# Computing NIQE features
				features = self.compute_NIQE_features(img=Y, block_size=niqe_block_size, C=1e-3)
				NIQE_features.append(features)

				# Space-Time MSCN statistics
				Y3d_mean = NSS_tools.spatiotemporal_filtering(img_buffer, self.temporal_filter_kernel)
				grad_Y3d_mscn = NSS_tools.spatiotemporal_filtering(grad_img_buffer, self.temporal_filter_kernel)

				# Space-Time Kurtosis features
				stats, grad_stats = NSS_tools.extract_kurtosis_statistics(Y3d_mean, grad_Y3d_mscn, self.space_time_length, center_y, center_x, self.R_sin, self.R_cos, self.theta)

				stats_arr = NSS_tools.unblockshaped(np.reshape(stats, (-1,self.space_time_length,self.space_time_length)), r1*self.space_time_length, r2*self.space_time_length)
				grad_stats_arr = NSS_tools.unblockshaped(np.reshape(grad_stats,(-1,self.space_time_length,self.space_time_length)), r1*self.space_time_length, r2*self.space_time_length)

				# Calculating Features
				features =  self.compute_BRISQUE_features(stats_arr)
				grad_features = self.compute_BRISQUE_features(grad_stats_arr)
				ST_chip_features.append(np.concatenate([features, grad_features], axis=0))

				# Resetting Buffers
				img_buffer = np.zeros((self.space_time_length, h, w))
				grad_img_buffer = np.zeros((self.space_time_length, h, w))


		return np.array(CGL_features), np.array(CGL_features_per_ST_chip), np.array(NIQE_features), np.array(ST_chip_features)


	def compute_score(self,
		video:np.array,
		scale_factor:float
	):
		# Features from Original Scale
		CGL_features1, CGL_features_per_ST_chip1, NIQE_features1, ST_chip_features1 = self.extract_features(video=video, niqe_block_size=(96,96))

		h,w = video.shape[1], video.shape[2]
		dsize = (int(scale_factor*h),int(scale_factor*w))

		downsampled_video = []
		for i in range(len(video)):
			downsampled_video.append(cv2.resize(video[i], (dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC))
		downsampled_video = np.asarray(downsampled_video)

		CGL_features2, CGL_features_per_ST_chip2, NIQE_features2, ST_chip_features2 = self.extract_features(video=downsampled_video, niqe_block_size=(48,48))

		# NIQE Score
		multi_scale_NIQE_features = np.concatenate((NIQE_features1, NIQE_features2), axis=-1)

		NIQE_features = []
		for i in range(len(NIQE_features1)):
			NIQE_features.append(self.compute_NIQE_score(multi_scale_NIQE_features[i]))


		# Appending features
		f_1_56 = np.concatenate((
			CGL_features1[:,0:4], CGL_features2[:,0:4],
			CGL_features1[:,4:8], CGL_features2[:,4:8],
			CGL_features1[:,8:24], CGL_features2[:,8:24],
			CGL_features1[:,24:28], CGL_features2[:,24:28]
		), axis=-1)

		f_57_112 = np.concatenate((
			CGL_features_per_ST_chip1[:,0:4], CGL_features_per_ST_chip2[:,0:4],
			CGL_features_per_ST_chip1[:,4:8], CGL_features_per_ST_chip2[:,4:8],
			CGL_features_per_ST_chip1[:,8:24], CGL_features_per_ST_chip2[:,8:24],
			CGL_features_per_ST_chip1[:,24:28], CGL_features_per_ST_chip2[:,24:28],
		), axis=-1)

		f_113_221 = np.concatenate((
			NIQE_features,
			ST_chip_features1[:, 0:18], ST_chip_features2[:, 0:18],
			ST_chip_features1[:, 18:36], ST_chip_features2[:, 18:36]
		), axis=-1)

		# ChipQA features
		X = np.concatenate((
			np.average(f_1_56, axis=0),
			np.average(f_57_112, axis=0),
			np.average(f_113_221, axis=0)
		), axis=0)

		return X
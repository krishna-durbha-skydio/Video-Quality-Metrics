"""
BRISQUE
"""
# Importing Libraries
import numpy as np
import scipy
import cv2
from numba import jit

import functions.NSS_tools as NSS_tools

class BRISQUE:
	def __init__(self,
		avg_window:np.array,
		C:float=1e-3
	):
		self.avg_window = avg_window
		self.C = self.C
		
	def compute_features(self, img:np.array):
		mscn, _, _ = NSS_tools.compute_image_mscn_transform(img, self.C, self.avg_window)

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
"""
Support functions for ChipQA.py
"""
# Importing Libraries
import numpy as np
import scipy
import math
import cv2
from scipy.special import gamma
from scipy.stats import skew, kurtosis
from numba import jit


# Precompuations
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
precomputed_gammas = a/(b*c)


# @jit(nopython=True)
def gen_gauss_window(
	lw:int,
	sigma:float
):
	"""
	Generate a Gaussian Window with zero mean and given sigma
	Args:
		lw (int): (2lw + 1) will be the length of window.
		sigma (float): Standard Deviation of Gaussian Distribution
	"""
	# Variance
	var = np.square(np.float32(sigma))

	# Weights
	lw = int(lw)
	weights = np.zeros((2 * lw + 1))
	weights[lw] = 1.0

	# Sum for Normalization
	sum = 1.0

	# Gaussian Distribution
	for ii in range(1, lw + 1):
		tmp = np.exp(-0.5 * np.float32(ii * ii) / var)
		weights[lw + ii] = tmp
		weights[lw - ii] = tmp
		sum += 2.0 * tmp

	# Normalization
	weights = weights/sum

	return weights


# @jit(nopython=True)
def spatiotemporal_filtering(
	img_buffer:np.array,
	avg_window:np.array,
	extend_mode='mirror'
):
	"""
	Calculating SpatioTemporal Mean
	Args:
		img_buffer (np.array): Image Buffer i.e List of frames
		avg_window (np.array): Window used for spatiotemporal filtering.
		extend_mode (str): Mode while filtering image buffer along temporal axis.
	"""
	filtered_img_buffer = np.zeros((img_buffer.shape))
	scipy.ndimage.correlate1d(img_buffer, avg_window, 0, filtered_img_buffer, mode=extend_mode)
	return filtered_img_buffer


# @jit(nopython=True)
def compute_gradient_magnitude(
	img:np.array
):
	"""
	Computing Gradient Magnitude
	Args:
		img_buffer (np.array): Image Buffer i.e List of frames
		avg_window (np.array): Window used for spatiotemporal filtering.
		extend_mode (str): Mode while filtering image buffer along temporal axis.
	"""
	gradient_x = cv2.Sobel(img, ddepth=-1, dx=1, dy=0)
	gradient_y = cv2.Sobel(img, ddepth=-1, dx=0, dy=1)
	gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)

	return gradient_mag


# @jit(nopython=True)
def paired_product(new_im):
	shift1 = np.roll(new_im.copy(), 1, axis=1)
	shift2 = np.roll(new_im.copy(), 1, axis=0)
	shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
	shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

	H_img = shift1 * new_im
	V_img = shift2 * new_im
	D1_img = shift3 * new_im
	D2_img = shift4 * new_im

	return (H_img, V_img, D1_img, D2_img) 


# @jit(nopython=True)
def compute_image_mscn_transform(
	image:np.array,
	C:float=1,
	avg_window:np.array=None,
	extend_mode='constant'
):
	"""
	Computing MSCN coefficients of an Image
	Args:
		(image): 2D Image
		C (int): Constant in denominator of MSCN calculation.
		avg_window (np.array): Window used as filter while MSCN calculation.
		extend_mode (str): Mode while filtering image.
	"""
	# Assertions
	assert len(np.shape(image)) == 2, "Image should be 2D"

	# Averaging Window
	if avg_window is None:
		avg_window = gen_gauss_window(3, 7.0/6.0)
	
	h, w = image.shape[0], image.shape[1]
	mu_image = np.zeros((h, w), dtype=np.float32)
	var_image = np.zeros((h, w), dtype=np.float32)
	image = np.array(image).astype('float32')
	
	# Calculating Mean
	scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
	scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)

	# Calculating Variance (Var(X) = E[X^2] - E[X]^2)
	scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
	scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
	var_image = np.sqrt(np.abs(var_image - mu_image**2))

	return (image - mu_image)/(var_image + C), var_image, mu_image


# @jit(nopython=True)
def estimate_ggd_features(
	vec:np.array
):
	"""
	Estimate GGD parameters using moment matching
	"""
	# Gamma Function values
	gam = np.asarray([x / 1000.0 for x in range(200, 10000, 1)])
	r_gam = (gamma(1.0/gam)*gamma(3.0/gam))/((gamma(2.0/gam))**2)
	
	# Beta Parameter
	sigma_square = np.mean(vec**2) # -(np.mean(vec))**2
	sigma = np.sqrt(sigma_square)
	E = np.mean(np.abs(vec))
	rho = sigma_square/(E**2+1e-6)

	# Getting the best possible alphaparam
	array_position = (np.abs(rho - r_gam)).argmin()
	alphaparam = gam[array_position]

	return alphaparam, sigma


# @jit(nopython=True)
def estimate_aggd_features(
	vec:np.array
):
	# Flattening Vector
	vec = vec.flatten()

	# Square of a vector
	vec_square = vec*vec

	# Left and Right data
	left_data = vec_square[vec<0]
	right_data = vec_square[vec>=0]

	# Left and Right Mean
	left_mean_sqrt = 0
	right_mean_sqrt = 0
	if len(left_data) > 0:
		left_mean_sqrt = np.sqrt(np.average(left_data))
	if len(right_data) > 0:
		right_mean_sqrt = np.sqrt(np.average(right_data))

	# Gamma_hat
	if right_mean_sqrt != 0:
		gamma_hat = left_mean_sqrt/right_mean_sqrt
	else:
		gamma_hat = np.inf

	# Solving Gamma-hat Norm
	vec_square_mean = np.mean(vec_square)
	if vec_square_mean != 0:
		r_hat = (np.average(np.abs(vec))**2) / (np.average(vec_square))
	else:
		r_hat = np.inf
	rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

	# Solve alpha by guessing values that minimize rhat_norm
	pos = np.argmin((precomputed_gammas - rhat_norm)**2)
	alpha = gamma_range[pos]

	# Gammas
	gam1 = scipy.special.gamma(1.0/alpha)
	gam2 = scipy.special.gamma(2.0/alpha)
	gam3 = scipy.special.gamma(3.0/alpha)

	# AGGD Ratio and Betas
	aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
	bl = aggdratio * left_mean_sqrt
	br = aggdratio * right_mean_sqrt

	# Mean parameter
	N = (br - bl) * (gam2/gam1)

	return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


# @jit(nopython=True)
def statistical_features(mscn):
	"""
	Estimating Statistical features from MSCN coefficients
	"""
	# alpha and sigma/beta after fitting GGD parameters
	alpha,sigma = estimate_ggd_features(mscn)

	# Skewness and Kurtosis
	skewness = skew(mscn.flatten())
	kurt = kurtosis(mscn.flatten())

	return alpha,sigma,skewness,kurt


# @jit(nopython=True)
def extract_kurtosis_slice(
	Y3d_mscn:np.array,
	cy:int,
	cx:int,
	R_sin:np.array,
	R_cos:np.array,
	theta:np.array,
	w:int,
	temporal_length:int
):
	"""
	Calculating kurtosis for a Space-Time Slice for input angles.

	Args:
		Y3d_mscn (np.array): 3D MSCN statistics
		cy (int): Row index to extract space-time cube.
		cx (int): Column index to extract space-time cube.
		R_sin (np.array): R_sin
		R_cos (np.array): R_cos
		theta (np.array): Angles of space-time slice planes.
		w (int): Width of frame.
		temporal_length (np.array): Temporal length of set of frames.
	"""
	# Kurtosis for each plane at different angles of Space-Time Cube
	space_time_kurtosis = np.zeros((len(theta),))
	data = np.zeros((len(theta), temporal_length**2))

	for index,_ in enumerate(theta):
		# Location of Space-Time Cube
		rsin_theta = R_sin[:,index]
		rcos_theta = R_cos[:,index]
		x_sts, y_sts = cx + rcos_theta, cy + rsin_theta

		# Get Space-Time Cube
		data[index,:] = Y3d_mscn[:, y_sts*w + x_sts].flatten() 

		# Calculate Kurtosis
		data_mean_power4 = np.mean((data[index,:] - np.mean(data[index,:]))**4)
		data_variance = np.var(data[index,:])
		space_time_kurtosis[index] = data_mean_power4/(data_variance**2 + 1e-4)

	# Selecting slice close to Gaussianity
	idx = (np.abs(space_time_kurtosis - 3)).argmin()
	data_slice = data[idx,:]

	return data_slice


# @jit(nopython=True)
def extract_kurtosis_statistics(
	img_buffer:np.array,
	grad_img_buffer:np.array,
	temporal_length:int,
	cy:np.array,
	cx:np.array,
	R_sin:np.array,
	R_cos:np.array,
	theta:np.array
):
	"""
	Extract Kurtosis Statistics of given set of frames.

	Args:
		img_buffer (np.array): Set of frames.
		grad_img_buffer (np.array): Gradient of the set of frames.
		temporal_length (np.array): Temporal length of set of frames.
		cy (np.array): Row indices to extract space-time cubes.
		cx (np.array): Column indices to extract space-time cubes.
		R_sin (np.array): R_sin
		R_cos (np.array): R_cos
		theta (np.array): Angles of space-time slice planes.
	"""
	# Spatial Dimensions
	h, w = img_buffer[temporal_length-1].shape[:2]

	# Flattening MSCN Statistics
	Y3d_mscn = np.reshape(img_buffer.copy(), (temporal_length, -1))
	grad_Y3d_mscn = np.reshape(grad_img_buffer.copy(), (temporal_length,-1))

	# Calculate Kurtosis Features
	kurtosis_features = [extract_kurtosis_slice(Y3d_mscn, cy[i], cx[i], R_sin, R_cos, theta, w, temporal_length) for i in range(len(cy))]
	grad_kurtosis_features = [extract_kurtosis_slice(grad_Y3d_mscn, cy[i], cx[i], R_sin, R_cos, theta, w, temporal_length) for i in range(len(cy))]

	return kurtosis_features, grad_kurtosis_features


# @jit(nopython=True)
def unblockshaped(arr, h, w):
	"""
	Unblocking a Blocked Array i.e If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
	then the returned array preserves the "physical" layout of the sublocks and has shape (h, w) where
	h * w = arr.size.
	"""
	n, nrows, ncols = arr.shape
	return (arr.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w))


# @jit(nopython=True)
def second_order_statistical_features(mscn):
	"""
	Second Order statistical features
	Args:
		mscn (np.array): MSCN coefficents
	"""
	# Paired Product
	pps1, pps2, pps3, pps4 = paired_product(mscn)

	# Fitting AGGD to paired products
	alpha1, N1, bl1, br1, lsq1, rsq1 = estimate_aggd_features(pps1)
	alpha2, N2, bl2, br2, lsq2, rsq2 = estimate_aggd_features(pps2)
	alpha3, N3, bl3, br3, lsq3, rsq3 = estimate_aggd_features(pps3)
	alpha4, N4, bl4, br4, lsq4, rsq4 = estimate_aggd_features(pps4)
	return np.array([
			alpha1, N1, lsq1**2, rsq1**2,	# (V)
			alpha2, N2, lsq2**2, rsq2**2,	# (H)
			alpha3, N3, lsq3**2, rsq3**2,	# (D1)
			alpha4, N4, lsq4**2, rsq4**2	# (D2)
		])


# @jit(nopython=True)
def extract_subband_features(
	mscn:np.array
):
	"""
	Extracting Subband Features
	Args:
		mscn (np.array): MSCN coefficents
		fit_AGGD_to_MSCN (bool): Fit AGGD to MSCN coefficients
	"""
	# Fitting AGGD to MSCN
	alpha_m, N, bl, br, lsq, rsq = estimate_aggd_features(mscn.copy())
	sigma = (bl+br)/2.0
	
	# Pairwise Product
	pps1, pps2, pps3, pps4 = paired_product(mscn)

	# Fitting AGGD to pairwise products
	alpha1, N1, bl1, br1, lsq1, rsq1 = estimate_aggd_features(pps1)
	alpha2, N2, bl2, br2, lsq2, rsq2 = estimate_aggd_features(pps2)
	alpha3, N3, bl3, br3, lsq3, rsq3 = estimate_aggd_features(pps3)
	alpha4, N4, bl4, br4, lsq4, rsq4 = estimate_aggd_features(pps4)

	return np.array([alpha_m, sigma]), np.array([
			alpha1, N1, bl1, br1,  # (V)
			alpha2, N2, bl2, br2,  # (H)
			alpha3, N3, bl3, br3,  # (D1)
			alpha4, N4, bl4, br4,  # (D2)
	])


# @jit(nopython=True)
def extract_on_patches(img, blocksizerow, blocksizecol):
	"""
	Extracting features on patches.
	"""
	h, w = img.shape
	blocksizerow = np.int16(blocksizerow)
	blocksizecol = np.int16(blocksizecol)

	patches = []
	for j in range(0, np.int16(h-blocksizerow+1), np.int16(blocksizerow)):
		for i in range(0, np.int16(w-blocksizecol+1), np.int16(blocksizecol)):
			patch = img[j:j+blocksizerow, i:i+blocksizecol]
			patches.append(patch)

	patches = np.array(patches)
	
	patch_features = []
	for p in patches:
		mscn_features, pp_features = extract_subband_features(p)
		patch_features.append(np.hstack((mscn_features, pp_features)))
	patch_features = np.array(patch_features)

	return patch_features


# @jit(nopython=True)
def extract_chroma_features(
	lab:np.array,
	C:float=1
):
	"""
	Extracting Chroma Features of a frame.
	Args:
		lab (np.array): lab components of the frame.
		C (int): Constant used in denominator for MSCN calculation.
	"""
	# Components
	a = lab[:,:,1]
	b = lab[:,:,2]

	# Chroma 
	chroma = np.sqrt(a**2+b**2)
	chroma_mscn,sigma_map,_ = compute_image_mscn_transform(chroma,C)
	sigma_mscn,_,_ = compute_image_mscn_transform(sigma_map,C)

	alpha,sigma,skewness,kurt = statistical_features(chroma_mscn)
	salpha,ssigma,sskewness,skurt = statistical_features(sigma_mscn)

	# First Order chroma features
	first_order_feats = np.asarray([alpha, sigma, skewness, kurt])
	first_order_feats_sigma_map = np.asarray([salpha, ssigma, sskewness, skurt])
	
	return first_order_feats, first_order_feats_sigma_map
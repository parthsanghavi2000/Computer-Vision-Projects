import numpy as np

def interp2(v, xq, yq):
	dim_input = 1
	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'

	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor < 0] = 0
	y_floor[y_floor < 0] = 0
	x_ceil[x_ceil < 0] = 0
	y_ceil[y_ceil < 0] = 0

	x_floor[x_floor >= w-1] = w-1
	y_floor[y_floor >= h-1] = h-1
	x_ceil[x_ceil >= w-1] = w-1
	y_ceil[y_ceil >= h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val


def edgeLink(M, Mag, Ori, low, high):
  img = np.copy(Mag)
  img[np.logical_not(M)] = 0

# Initial strong edge map
  strong_edge = img >= high

# Initialize two array to start the while loop
  pre_uncertain = np.array([1])
  curr_uncertain = np.array([0])

  while not np.array_equal(pre_uncertain, curr_uncertain):
    # Uncertain edge
    uncertain = np.logical_and(img >= low, img < high)
    pre_uncertain = np.copy(uncertain)

    # Find neighbors coordinate 
    nr, nc = Mag.shape[0], Mag.shape[1]
    x, y = np.meshgrid(np.arange(nc), np.arange(nr))
    x_next= np.clip(x + np.cos(Ori + np.pi/2), 0, nc - 1)
    y_next = np.clip(y + np.sin(Ori + np.pi/2), 0, nr - 1)

    x_next2 = np.clip(x + np.cos(Ori - np.pi/2), 0, nc - 1)
    y_next2 = np.clip(y + np.sin(Ori - np.pi/2), 0, nr - 1)

    # Interpolate to get each pixel's neighbor
    neighbor1 = interp2(img, x_next, y_next)
    neighbor2 = interp2(img, x_next2, y_next2)

    # Set neighbor value out of bound to zero
    small = 1e-10
    indice1 = (x + np.cos(Ori + np.pi/2) > nc - 1 + small) | (x + np.cos(Ori + np.pi/2) < -small) | (y + np.sin(Ori + np.pi/2) > nr - 1 + small) | (y + np.sin(Ori + np.pi/2) < -small)
    neighbor1[indice1] = 0
    indice2 = (x + np.cos(Ori - np.pi/2) > nc - 1 + small) | (x + np.cos(Ori - np.pi/2) < -small) | (y + np.sin(Ori - np.pi/2) > nr - 1 + small) | (y + np.sin(Ori - np.pi/2) < -small)
    neighbor2[indice2] = 0

    strong_neighbor = (neighbor1 >= high) | (neighbor2 >= high)
    # Weak pixels in uncertain that should be changed to strong
    toUpDate = np.logical_and(uncertain, strong_neighbor)
    # Add those new strong edges to existing strong edge
    strong_edge = np.logical_or(strong_edge, toUpDate)
    # Change new strong edge corresponding magnitude to largest neighbor magnitude
    img = np.where(toUpDate, np.maximum(neighbor1, neighbor2), img)
    # Calculate the uncertain map after updating new strong edge and compare it with previous uncertain map
    uncertain = np.logical_and(img >= low, img < high)
    curr_uncertain = np.copy(uncertain)

  E = strong_edge

  return E
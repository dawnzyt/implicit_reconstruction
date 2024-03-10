# 
import numpy as np
import torch

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
            
# adatpted from https://github.com/dakshaau/ICP/blob/master/icp.py#L4 for rotation only 
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    AA = A
    BB = B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    return R
def align_x(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[1] == depth2.shape[1]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, :, s2:e2], depth1[:, :, s1:e1], torch.ones_like(depth1[:, :, s1:e1]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1], depth1.shape[2] + depth2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = depth1[:, :, :s1]
    result[:, :, depth1.shape[2]:] = depth2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    result[:, :, s1:depth1.shape[2]] = depth1[:, :, s1:] * weight + depth2_aligned[:, :, :e2] * (1 - weight)

    return result

# align depth map in the y direction from top to down
def align_y(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[2] == depth2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, s2:e2, :], depth1[:, s1:e1, :], torch.ones_like(depth1[:, s1:e1, :]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1] + depth2.shape[1] - (e1 - s1), depth1.shape[2]))

    result[:, :s1, :] = depth1[:, :s1, :]
    result[:, depth1.shape[1]:, :] = depth2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1-s1))[None, :, None]
    result[:, s1:depth1.shape[1], :] = depth1[:, s1:, :] * weight + depth2_aligned[:, :e2, :] * (1 - weight)

    return result

def align_normal_x(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[1] == normal2.shape[1]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, :, s2:e2].reshape(3, -1).T, normal1[:, :, s1:e1].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1], normal1.shape[2] + normal2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = normal1[:, :, :s1]
    result[:, :, normal1.shape[2]:] = normal2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    
    result[:, :, s1:normal1.shape[2]] = normal1[:, :, s1:] * weight + normal2_aligned[:, :, :e2] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

# align normal map in the y direction from top to down
def align_normal_y(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[2] == normal2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, s2:e2, :].reshape(3, -1).T, normal1[:, s1:e1, :].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1] + normal2.shape[1] - (e1 - s1), normal1.shape[2]))

    result[:, :s1, :] = normal1[:, :s1, :]
    result[:, normal1.shape[1]:, :] = normal2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1-s1))[None, :, None]
    
    result[:, s1:normal1.shape[1], :] = normal1[:, s1:, :] * weight + normal2_aligned[:, :e2, :] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)):int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


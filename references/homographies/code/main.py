import json
from scipy import signal
import skimage.io as skio
import numpy as np
from scipy.linalg import lstsq, solve
from scipy.interpolate import RegularGridInterpolator
import cv2
import skimage as sk

def compute_H(pts1, pts2):
    M = np.array([], dtype=np.int64).reshape(0,8)
    res_coords = np.array(pts2).flatten()
    res_coords = res_coords.reshape(res_coords.shape[0], 1)
    for (x, y), (x_2, y_2) in zip(pts1, pts2):
        row1 = np.array([x, y, 1, 0, 0, 0, -x*x_2, -y*x_2])
        row2 = np.array([0, 0, 0, x, y, 1, -x*y_2, -y*y_2])
        M = np.vstack((M, row1, row2))
    
    lsq = lstsq(M, res_coords)[0]
    #lsq = solve(M, res_coords)
    return np.append(lsq, 1).reshape(3,3)

def bilinear_interpolation(image, coords): 
    y, x = np.arange(image.shape[0]), np.arange(image.shape[1])
    interp = RegularGridInterpolator((y, x), image, method='linear', bounds_error=False, fill_value=0)
    return interp(np.array([coords[1, :], coords[0, :]]).T)
   
def get_image_corners(im, H):
    print(im.shape)
    rectangle_corners = np.array([np.array([0,0]), np.array([0, im.shape[0]-1]), np.array([im.shape[1]-1, im.shape[0]-1]), np.array([im.shape[1]-1, 0])]).T
    rectangle_corners = np.vstack([rectangle_corners, np.ones(4)])
    corners = H @ rectangle_corners
    return (corners[:2]*(1/corners[2])).T

def warp_image(image, H):
    warp_corners = get_image_corners(image, H)
    (x, y, width, height) = cv2.boundingRect(np.float32(warp_corners))
    rr, cc = sk.draw.polygon(warp_corners[:, 1] - y, warp_corners[:, 0] - x)
    rr, cc = rr + y, cc + x
    points_vector = np.vstack((cc, rr, np.ones(len(rr))))
    mat_source = np.linalg.inv(H) @ points_vector
    interpolated_values = bilinear_interpolation(image, mat_source[:2, :] * 1/mat_source[2])
    destination_image = np.zeros((height, width, 3), dtype=np.float32)
    destination_image[rr - y, cc - x] = interpolated_values
    mask = np.zeros_like(destination_image[:, :, 0])
    mask[rr - y, cc - x] = 255
    return destination_image, y, x, warp_corners, mask

def naive_join(images, offsets_y, offsets_x):
    offsets_y = [offset - min(offsets_y) for offset in offsets_y]
    offsets_x = [offset - min(offsets_x) for offset in offsets_x]
    destination_image = np.zeros((max([offsets_y[i] + images[i].shape[0] for i in range(len(offsets_x))]), max([offsets_x[i] + images[i].shape[1] for i in range(len(offsets_x))]), 3), dtype=np.float32)

    for image, offset_y, offset_x in zip(images, offsets_y, offsets_x):
        destination_image[offset_y:image.shape[0]+offset_y, offset_x:image.shape[1]+offset_x][destination_image[offset_y:image.shape[0]+offset_y, offset_x:image.shape[1]+offset_x]==0] = image[destination_image[offset_y:image.shape[0]+offset_y, offset_x:image.shape[1]+offset_x]==0]
    print('good', destination_image.shape)
    return destination_image

def get_full_size_separate_images(images, masks, offsets_y, offsets_x):
    offsets_y = [offset - min(offsets_y) for offset in offsets_y]
    offsets_x = [offset - min(offsets_x) for offset in offsets_x]
    dest_images = []
    dest_masks = []

    for image, mask, offset_y, offset_x in zip(images, masks, offsets_y, offsets_x):
        destination_image = np.zeros((max([offsets_y[i] + images[i].shape[0] for i in range(len(offsets_x))]), max([offsets_x[i] + images[i].shape[1] for i in range(len(offsets_x))]), 3), dtype=np.float32)
        destination_mask = np.zeros_like(destination_image[:, :, 0])
        destination_image[offset_y:image.shape[0]+offset_y, offset_x:image.shape[1]+offset_x] = image
        dest_images.append(destination_image)
        destination_mask[offset_y:image.shape[0]+offset_y, offset_x:image.shape[1]+offset_x] = mask
        dest_masks.append(destination_mask)
    return dest_images, dest_masks
    
im1_name = 'elevator_1.png'
im2_name = 'elevator_2.png'
# im3_name = 'great_hall_3.jpg'

im1 = skio.imread('photos/'+ im1_name)[:, :, :3]
im2 = skio.imread('photos/'+ im2_name)[:, :, :3]
# im3 = skio.imread('photos/'+ im3_name)

with open('correspondances/elevator_1_elevator_2.json') as f:
    data = json.load(f)

pts1_1, pts1_2 = np.array(data['im1Points']), np.array(data['im2Points']) 

# with open('correspondances/great_hall_2_great_hall_3.json') as f:
#     data = json.load(f)

# pts2_1, pts2_2 = np.array(data['im1Points']), np.array(data['im2Points']) 

H_1 = compute_H(pts1_1, pts1_2)
im1_warp, offset_y, offset_x, warp_corners, mask_1 = warp_image(im1, H_1)

images = [im1_warp, im2]
masks = [mask_1, 255*np.ones_like(im2[:, :, 0])]
offsets_y = [offset_y, 0]
offsets_x = [offset_x, 0]
joined_2 = naive_join(images, offsets_y, offsets_x)

full_images, full_masks = get_full_size_separate_images(images, masks, offsets_y, offsets_x)

result = np.zeros(full_images[0].shape, dtype=np.float32)
weight_sum = np.zeros(full_images[0][:, :, 0].shape, dtype=np.float32)

for warped, mask in zip(full_images, full_masks):
    result += warped * mask[..., np.newaxis]
    weight_sum += mask

result = np.divide(result, weight_sum[..., np.newaxis], where=weight_sum[..., np.newaxis]!=0)

skio.imsave('im1warp.jpg', np.clip(im1_warp, 0, 255).astype(np.uint8))
skio.imsave('im12warped.jpg', np.clip(joined_2, 0, 255).astype(np.uint8))
skio.imsave('im1warpedlarge.jpg', np.clip(full_images[0], 0, 255).astype(np.uint8))
skio.imsave('im2large.jpg', np.clip(full_images[1], 0, 255).astype(np.uint8))
# skio.imsave('mask.jpg', np.clip(mask, 0, 255).astype(np.uint8))
skio.imsave('finalmosaic.jpg', np.clip(result, 0, 255).astype(np.uint8))


#rect = skio.imread('photos/'+ 'board.jpg')
# rect = skio.imread('photos/'+ 'window.jpg')

# with open('correspondances/window.json') as f:
#     data = json.load(f)

# pts_board = np.array(data['im1Points'])
# H_board = compute_H(pts_board, np.array([[50, 50], [400, 50], [50, 600], [400, 600]]))
# rect_warp, _,_ = warp_image(rect, H_board)

# print(rect_warp.shape)

# skio.imsave('boardwarp.jpg', np.clip(rect_warp, 0, 255).astype(np.uint8))
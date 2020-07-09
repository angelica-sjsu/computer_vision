from numpy import *
from matplotlib import *
from pylab import *
from PIL import Image as Im
from scipy.ndimage import filters
import os


def get_image(path):
    '''
    :param path: image path
    :return: retrieved image
    '''
    img = Im.open(path)
    return img


def show_contour(img):
    '''
    Design the contour of the images
    :param img: image to use for contour
    :return: None
    '''
    figure()
    gray()
    contour(img, origin='image')
    axis('equal')
    axis('off')
    show()


def gaussian_filter(img, sigma=2):
    '''
    Blurs the image based on sigma
    :param img: base image
    :param sigma: standard deviation for the Gaussian filter, determines blur level
    :return: Blurred image and array
    '''
    img_array = array(img)
    img = filters.gaussian_filter(img, sigma)
    imshow(Im.fromarray(img))
    return Im.fromarray(img), img


def unsharp_masking(img, sigma=2):
    '''
    Applies the difference between original and blured image
    :param img: true image
    :param sigma: value to tweak the filter
    :return: array of the resulting image difference
    '''
    blurred = filters.gaussian_filter(img, sigma)
    unsharp_mask = img - blurred
    Im.fromarray(unsharp_mask).save(os.path.join('images', 'unsharped.jpg'))
    return unsharp_mask


def quotient_image(img):
    '''
    Applies the quotient between img and the blurred image
    :param img: baseline image
    :return: image version of the quotient image and its array
    '''
    _, filtered = gaussian_filter(img)
    quotient = array(divide(img,filtered), 'uint8')
    return Im.fromarray(quotient), quotient


def image_gradients(img, sigma=2):
    '''
    Finds outline of images using gradients.
    Applying gaussian filter
    :param img: basline image
    :return: derivatives in x and y directions
    '''
    dv_x = zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma),(0, 1),dv_x)
    dv_y = zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), dv_y)
    return dv_x, dv_y


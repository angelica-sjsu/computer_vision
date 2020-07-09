from tools import *
import matplotlib.pyplot as plt


def ex1(img):
    gray = img.convert('L')
    imshow(gray)
    show_contour(gray)
    f1_img, f1 = gaussian_filter(gray, 2)
    show_contour(f1)


def ex2(img):
    unsharped = unsharp_masking(img)
    plt.imshow(unsharped)
    plt.show()


def ex3(img):
    img_q, q = quotient_image(img)
    plt.imshow(img_q)
    plt.show()


def ex4(img):
    dv_x, dv_y = image_gradients(array(img.convert('L')), sigma=2)
    _, axs = plt.subplots(4, 1, figsize=(20, 20))
    axs = axs.flatten()
    magnitude = sqrt(dv_x**2 + dv_y**2)
    for img, ax in zip([dv_x, img, dv_y, magnitude], axs):
        ax.imshow(img)
    plt.show()


def entry():
    # print('This is entry')
    img = get_image('sample.jpg')
    img2 = get_image('black_rot.JPG')
    #ex1(img)
    #ex2(img)
    #ex3(array(img.convert('L')))
    ex4(img)
    ex4(img2)


if __name__ == '__main__':
    entry()
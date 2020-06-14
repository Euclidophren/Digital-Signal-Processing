# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.image as mpimg
# from scipy.signal import convolve2d as conv
# from skimage import color, data, restoration, img_as_float
#
#
# def main():
#     img = mpimg.imread('Lab1_3_5.bmp')
#
#     image = img_as_float(img) / 255
#     gray = color.rgb2gray(image)
#     psf = np.ones((5, 5)) / 25
#     gray = conv2(gray, psf, 'same')
#
#     # Restore Image using Richardson-Lucy algorithm
#     deconvolved_rl = restoration.richardson_lucy(gray,
#                                                  psf, iterations=50)
#
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
#     plt.gray()
#     ax[0].imshow(img)
#     ax[0].set_title('Исходное изображение')
#
#     ax[1].imshow(deconvolved_rl)
#     ax[1].set_title('Восстановленное')
#
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

import oct2py

octave = oct2py.Oct2Py()
script = " I=double(imread('Lab1_3_5.bmp')) / 255; " \
    "figure; imshow(I); title('Source image');" \
    "PSF=fspecial('motion', 54, 65); "\
    "[J1 P1]=deconvblind(I, PSF);"\
    "figure; imshow(J1); title('Recovered image');"

with open("myScript.m", "w+") as f:
    f.write(script)

octave.myScript(7)
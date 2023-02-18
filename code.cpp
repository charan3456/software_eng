

import cv2

import numpy as np

from matplotlib import pyplot as plt

path = r'/Users//Desktop/DIP LAB/Image.jpeg'

img=cv2.imread(path) #Reading An images

cv2.imshow("Original_image",img) #Display Image

cv2.waitKey(0)





resize_img=cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1) #Resizing Images

cv2.imshow("Resized_image",resize_img) #Display Image

cv2.waitKey(0)









gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting  into grayscale

rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Converting bgr to RGB image

cv2.imshow("Gray",gray)

cv2.waitKey(0)

cv2.imshow("RGB",rgb)

cv2.waitKey(0)



#rgbpanes

def split_rgb():

  cv2.imshow('Image',img)

  cv2.waitKey(0)

  b,g,r=cv2.split(img)

  cv2.imshow('b',b)

  cv2.waitKey(0)

  cv2.imshow('g',g)

  cv2.waitKey(0)

  cv2.imshow('r',r)

  cv2.waitKey(0)

split_rgb()



import matplotlib.pyplot as mp

import numpy as np

from PIL import Image

# Creating the 144 X 144 NumPy Array with random values

arr = np.random.randint(0,255, size=(256, 256),dtype=np.uint8)

i=0

j=0

while(i<=255):

  j=0

  while(j<=255):

      arr[i][j]=i

    

      j+=1

  i+=1

# Converting the NumPy Array into an image

img = Image.fromarray(arr)

img.save("file.png")



mp.imshow(img)





cv2.destroyAllWindows()



###############################################################################################





import cv2

import numpy as np

from matplotlib import pyplot as plt

path = r'/Users//Desktop/DIP LAB/Satya.jpeg'

img=cv2.imread(path) # Reading An image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting  into grayscale

#Negative image

neg_img=  255 - gray

cv2.imshow("neg_img",neg_img)

cv2.waitKey(0)

#Flip Image

flip_img=cv2.flip(gray,0)

cv2.imshow("flip_img",flip_img)

cv2.waitKey(0)

#Contrast strech

xp = [0, 64, 128, 192, 255]

fp = [0, 16, 128, 240, 255]

x = np.arange(256)

table = np.interp(x, xp, fp).astype('uint8')

image = cv2.LUT(gray, table)

cv2.imshow("contrast", image)

cv2.waitKey(0)

#Thresholding

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)

ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)

ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','Binary','Binary_Inversion','Trunc','ToZero','ToZero_Inv']

images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):

  plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)

  plt.title(titles[i])

  plt.xticks([]),plt.yticks([])

plt.show()



###############################################################################################



# Python program to illustrate

# arithmetic operation of

# addition of two images



# organizing imports

from ctypes import sizeof

import cv2

import numpy as np



# path to input images are specified and

# images are loaded with imread command

image1 = cv2.imread('/Users//Desktop/DIP LAB/LAB2 Final/sky.jpg')

image2 = cv2.imread('/Users//Desktop/DIP LAB/LAB2 Final/road.jpg')

print(image1.shape)

print(image2.shape)

image1 = image1[0:2773,0:4095]

print(image1.shape)





#Addition

weightedSum = cv2.addWeighted(image1, 1, image2, 1, 0)

cv2.imshow('Weighted Image', weightedSum)

npsum = image1+image2

cv2.imshow("NPSUM",npsum)



#Subtraction

subcv=cv2.subtract(image1,image2)

cv2.imshow("Subtracted Image",subcv)

subnp=image1 - image2

cv2.imshow("NPSubtraction",subnp)





#Multiplication

multcv=cv2.multiply(image1,image2)

cv2.imshow("Multiply",multcv)

multnp=image1*image2

cv2.imshow("NPMultiply",multnp)



# Divison

divcv=cv2.divide(image2,image1)

cv2.imshow("Divison",divcv)

divnp=image2/image1

cv2.imshow("NPDivison",divnp)

cv2.waitKey(0)



###############################################################################################

# histo equilization

import cv2

import matplotlib.pyplot as plt

import numpy as np

path=r"/Users//Desktop/DIP LAB/LAB2 Final/lowcontrast.jpg"

img = cv2.imread(path)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()

cdf_normalized = cdf * hist.max()/ cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)

cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])

cdf2 = hist2.cumsum()

cdf_normalized2 = cdf2 * hist2.max()/ cdf2.max()

imgs = [img, img2]

title = ['original', 'stretched image', 'histo', 'stretch_histo']

plt.subplot(2,2,1)

plt.title(title[0])

plt.imshow(imgs[0])

plt.xticks([])

plt.yticks([])

plt.subplot(2,2,2)

plt.title(title[1])

plt.imshow(imgs[1])

plt.xticks([])

plt.yticks([])

plt.subplot(2, 2, 3)

plt.title(title[2])

plt.plot(cdf_normalized, color = 'b')

plt.hist(img.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'center right')

plt.subplot(2, 2, 4)

plt.title(title[3])

plt.plot(cdf_normalized2, color = 'b')

plt.hist(img2.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'center right')

plt.show()



###############################################################################################



#histo matplotlib

from turtle import color

import cv2

import matplotlib.pyplot as plt

img = cv2.imread("/Users//Desktop/DIP LAB/LAB2 Final/lowcontrast.jpg")

cv2.imshow("Image",img)

histg1 = cv2.calcHist([img],[0],None,[256],[0,256])

histg2 = cv2.calcHist([img],[1],None,[256],[0,256])

histg3 = cv2.calcHist([img],[2],None,[256],[0,256])

plt.plot(histg1,color="b")

plt.plot(histg2,color="g")

plt.plot(histg3,color="r")

plt.title("Color Histogram")

plt.xlabel("Color value")

plt.ylabel("Pixel count")

plt.show()

cv2.waitKey(0)





###############################################################################################



#histo munpy

from turtle import color

import cv2

import numpy as np

import matplotlib.pyplot as plt

img = cv2.imread("/Users/

/Desktop/DIP LAB/LAB2 Final/lowcontrast.jpg")

cv2.imshow("Image",img)



hist,bins = np.histogram(img.ravel(),256,[0,256])

#show the plotting graph of an image

plt.plot(hist)

plt.show()

plt.waitforbuttonpress()





###############################################################################################





#histo underexposed

from turtle import color

import cv2

import matplotlib.pyplot as plt

import numpy as np

path=r"/Users/

/Desktop/DIP LAB/LAB2 Final/underexposed.jpg"

img = cv2.imread(path)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()

cdf_normalized = cdf * hist.max()/ cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)

cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])

cdf2 = hist2.cumsum()

cdf_normalized2 = cdf2 * hist2.max()/ cdf2.max()

imgs = [img, img2]

title = ['original', 'stretched image', 'histo', 'stretch_histo']

plt.subplot(2,2,1)

plt.title(title[0])

plt.imshow(imgs[0])

plt.xticks([])

plt.yticks([])

plt.subplot(2,2,2)

plt.title(title[1])

plt.imshow(imgs[1])

plt.xticks([])

plt.yticks([])

plt.subplot(2, 2, 3)

plt.title(title[2])

plt.plot(cdf_normalized, color = 'b')

plt.hist(img.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'center right')

plt.subplot(2, 2, 4)

plt.title(title[3])

plt.plot(cdf_normalized2, color = 'b')

plt.hist(img2.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'center right')

plt.show()



###############################################################################################



#histo with mask

import cv2

import numpy as np

import matplotlib.pyplot as plt



img_1 = cv2.imread("/Users/

/Desktop/DIP LAB/LAB2 Final/lowcontrast.jpg")

img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)

cv2.imshow("Image",img_1)



mask = np.zeros(img_1.shape[:2], np.uint8)

mask[0:2000, 1000:2000] = 255

masked_img = cv2.bitwise_and(img_1,img_1,mask = mask)

#Calculate histogram with mask and without mask

hist_full = cv2.calcHist([img_1],[0],None,[256],[0,256])

hist_mask = cv2.calcHist([img_1],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img_1, 'gray')

plt.subplot(222), plt.imshow(mask,'gray')

plt.subplot(223), plt.imshow(masked_img, 'gray')

plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)

plt.xlim([0,256])

plt.show()







###############################################################################################



#and

import cv2







img1 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image1.jpg",0)

img2 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image2.jpg",0)

print(img1.shape)

img1 = img1[900:1800,800:1700]

print(img1.shape)

print(img2.shape)

bitwiseand = cv2.bitwise_and(img2,img1,mask=None)

cv2.imshow("BITWISE AND",bitwiseand)

cv2.waitKey(0)



###############################################################################################





#fourier

import cv2 as cv

import cv2

import numpy as np

from matplotlib import pyplot as plt

from numpy import double

from scipy import fftpack, ndimage



path = r'/Users/

/Documents/DIP LAB/LAB 4/underexposed.jpeg'



val_img = cv.imread(path)

gray = cv.cvtColor(val_img, cv.COLOR_BGR2GRAY)  # converting  into grayscale

val_img = cv2.cvtColor(val_img,cv2.COLOR_BGR2RGB)



dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))



plt.subplot(121), plt.imshow(val_img, cmap='gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()

# --------Low Pass Filter --------#

kernel = np.array([[1, 1, 1, 1, 1],

                 [1, 1, 1, 1, 1],

                 [1, 1, 1, 1, 1],

                 [1, 1, 1, 1, 1],

                [1, 1, 1, 1, 1]])

kernel = kernel / sum(kernel)



# filter the source image

img_rst = cv.filter2D(val_img, -1, kernel)



# save result image

cv.imshow('result.jpg', img_rst)

# -------------- Applying Various High Pass Filters------------#

# apply laplacian blur

laplacian = cv.Laplacian(gray, cv.CV_64F)



# sobel x filter where dx=1 and dy=0

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=7)



# sobel y filter where dx=0 and dy=1

sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=7)



# combine sobel x and y

sobel = cv.bitwise_and(sobelx, sobely)



#plot images

plt.subplot(2, 2, 1)

plt.imshow(laplacian, cmap='gray')

plt.title('Laplacian')



plt.subplot(2, 2, 2)

plt.imshow(sobelx, cmap='gray')

plt.title('SobelX')



plt.subplot(2, 2, 3)

plt.imshow(sobely, cmap='gray')

plt.title('SobelY')



plt.subplot(2, 2, 4)

plt.imshow(sobel, cmap='gray')

plt.title('Sobel')



plt.show()

# --------------Applying Low Pass Filter and Using IFFT to reconstruct image-------#

img_float32 = np.float32(gray)

dft = cv.dft(img_float32, flags=cv.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)



rows, cols = gray.shape

crow, ccol = rows / 2, cols / 2  # center



#create a mask first, center square is 1, remaining all zeros

mask = np.zeros((rows, cols, 2), np.uint8)

mask[int(crow - 30): int(crow + 30),int(ccol - 30): int(ccol + 30)]= 1



# apply mask and inverse DFT

fshift = dft_shift * mask

f_ishift = np.fft.ifftshift(fshift)

img_back = cv.idft(f_ishift)

img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])



plt.subplot(121), plt.imshow(gray, cmap='gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(img_back, cmap='gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])



plt.show()





###############################################################################################



#intersection

import cv2







img1 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image3.jpg",0)

img2 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image4.jpg",0)

img1 = img1[0:626,0:626]

bitwiseand = cv2.bitwise_and(img1,img2,mask=None)

bitwiseor = cv2.bitwise_or(img1,img2,mask=None)

bitwisexor = cv2.bitwise_xor(img1,img2,mask=None)



cv2.imshow("BITWISE AND",bitwiseand)

cv2.imshow("BITWISE XOR",bitwisexor)

cv2.imshow("BITWISE OR",bitwiseor)

cv2.waitKey(0)





###############################################################################################



#not

import cv2







img1 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image1.jpg",0)

img2 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image2.jpg",0)

print(img1.shape)

img1 = img1[900:1800,800:1700]

print(img1.shape)

print(img2.shape)

bitwisenot = cv2.bitwise_not(img2,mask=None)

cv2.imshow("BITWISE NOT",bitwisenot)

cv2.waitKey(0)







###############################################################################################

#or

import cv2







img1 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image1.jpg",0)

img2 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image2.jpg",0)

print(img1.shape)

img1 = img1[900:1800,800:1700]

print(img1.shape)

print(img2.shape)

bitwiseor = cv2.bitwise_or(img2,img1,mask=None)

cv2.imshow("BITWISE OR",bitwiseor)

cv2.waitKey(0)



###############################################################################################

#xor

import cv2







img1 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image1.jpg",0)

img2 = cv2.imread("/Users/

/Documents/DIP LAB/LAB 4/image2.jpg",0)

print(img1.shape)

img1 = img1[900:1800,800:1700]

print(img1.shape)

print(img2.shape)

bitwisexor = cv2.bitwise_xor(img2,img1,mask=None)

cv2.imshow("BITWISE XOR",bitwisexor)

cv2.waitKey(0)



###############################################################################################

# gaussian

import matplotlib

import numpy as np

import cv2

import matplotlib.pyplot as plt

from PIL import Image,ImageFilter



image = cv2.imread("/Users/

/Documents/DIP LAB/LAB 5/underexposed.jpeg")

image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

figure_size = 9

new_image = cv2.GaussianBlur(image,(figure_size,figure_size),0)

plt.figure(figsize=(12,6))

plt.subplot(121)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_HSV2RGB))

plt.title('Original')

plt.xticks([])

plt.yticks([])

plt.subplot(122)

plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB))

plt.title('Gaussian Filter')

plt.xticks([])

plt.yticks([])

plt.show()



###############################################################################################

# high pass

from configparser import Interpolation

import cv2

image = cv2.imread("/Users/

/Documents/DIP LAB/LAB 5/underexposed.jpeg")

image = cv2.resize(image,(1000,657),interpolation=cv2.INTER_CUBIC)

filtered = image-cv2.GaussianBlur(image,(21,21),3)+127



cv2.imshow("Original",image)

cv2.imshow("High Passed Filter",filtered)

cv2.waitKey(0)









###############################################################################################

#homomorphic

import logging

import numpy as np



# Homomorphic filter class

class HomomorphicFilter:



  def __init__(self, a = 0.5, b = 1.5):

      self.a = float(a)

      self.b = float(b)



  # Filters

  def __butterworth_filter(self, I_shape, filter_params):

      P = I_shape[0]/2

      Q = I_shape[1]/2

      U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')

      Duv = (((U-P)**2+(V-Q)**2)).astype(float)

      H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])

      return (1 - H)



  def __gaussian_filter(self, I_shape, filter_params):

      P = I_shape[0]/2

      Q = I_shape[1]/2

      H = np.zeros(I_shape)

      U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')

      Duv = (((U-P)**2+(V-Q)**2)).astype(float)

      H = np.exp((-Duv/(2*(filter_params[0])**2)))

      return (1 - H)



  # Methods

  def __apply_filter(self, I, H):

      H = np.fft.fftshift(H)

      I_filtered = (self.a + self.b*H)*I

      return I_filtered



  def filter(self, I, filter_params, filter='butterworth', H = None):



      #  Validating image

      if len(I.shape) != 2:

          raise Exception('Improper image')



      # Take the image to log domain and then to frequency domain

      I_log = np.log1p(np.array(I, dtype="float"))

      I_fft = np.fft.fft2(I_log)



      # Filters

      if filter=='butterworth':

          H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)

      elif filter=='gaussian':

          H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)

      elif filter=='external':

          print('external')

          if len(H.shape) != 2:

              raise Exception('Invalid external filter')

      else:

          raise Exception('Selected filter not implemented')

    

      # Apply filter on frequency domain then take the image back to spatial domain

      I_fft_filt = self.__apply_filter(I = I_fft, H = H)

      I_filt = np.fft.ifft2(I_fft_filt)

      I = np.exp(np.real(I_filt))-1

      return np.uint8(I)

# End of class HomomorphicFilter



if __name__ == "__main__":

  import cv2



  # Main code

  img = cv2.imread("/Users/

  /Documents/DIP LAB/LAB 5/underexposed.jpeg")[:, :, 0]

  homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)

  img_filtered = homo_filter.filter(I=img, filter_params=[30,2])

  cv2.imshow("Filtered Image",img_filtered)

  cv2.waitKey()







###############################################################################################

#mean

import matplotlib

import numpy as np

import cv2

import matplotlib.pyplot as plt

from PIL import Image,ImageFilter



image = cv2.imread("/Users/

/Documents/DIP LAB/LAB 5/underexposed.jpeg")

image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

figure_size = 9

new_image = cv2.blur(image,(figure_size,figure_size))

plt.figure(figsize=(12,6))

plt.subplot(121)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_HSV2RGB))

plt.title('Original')

plt.xticks([])

plt.yticks([])

plt.subplot(122)

plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB))

plt.title('Mean Filter')

plt.xticks([])

plt.yticks([])

plt.show()









###############################################################################################

#median

import matplotlib

import numpy as np

import cv2

import matplotlib.pyplot as plt

from PIL import Image,ImageFilter



image = cv2.imread("/Users/

/Documents/DIP LAB/LAB 5/underexposed.jpeg")

image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

figure_size = 9

new_image = cv2.medianBlur(image,figure_size)

plt.figure(figsize=(12,6))

plt.subplot(121)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_HSV2RGB))

plt.title('Original')

plt.xticks([])

plt.yticks([])

plt.subplot(122)

plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB))

plt.title('Median Filter')

plt.xticks([])

plt.yticks([])

plt.show()



###############################################################################################

#lab6

#translation zoom

import cv2 as cv

import numpy as np

import numpy as np

from scipy import ndimage

from matplotlib import pyplot as plt



path_1 = r'/Users/

/Documents/DIP LAB/LAB 6/underexposed.jpeg'

img_1 = cv.imread(path_1)



gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)





# -----------------------------Translation-----------------------#

# Store height and width of the image

height, width = img_1.shape[:2]

quarter_height, quarter_width = height / 4, width / 4



T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

img_translation = cv.warpAffine(img_1, T, (width, height))





cv.imshow("Translation", img_translation)





# ---------------Zooming-------------#

def zoom(img, zoom_factor=1.5):

 return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv.INTER_LINEAR)





zoom_img = zoom(img_1)

cv.imshow("Zooming", zoom_img)





# ---------------------Shrinking------------#

def shrink(img, shrink_factor=1 / 3):

  return cv.resize(img, None, fx=shrink_factor, fy=shrink_factor, interpolation=cv.INTER_AREA)





shrink_img = shrink(img_1)

cv.imshow("Shrink", shrink_img)



# --------------Scaling------------------#

img_scaled = cv.resize(img_1, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)

cv.imshow("Scaling", img_scaled)



# -----------------Rotating--------------#

rotated_img = cv.rotate(img_1, cv.ROTATE_90_CLOCKWISE)

cv.imshow("Rotation", rotated_img)



# 2

img_np = np.array(img_1, dtype=float)

#---------------High Pass Filter-------------#

# A very simple and very narrow highpass filter

kernel = np.array([[-1, -1, -1],

                 [-1, 5, -1],

                 [-1, -1, -1]])



highpass_3x3 = cv.filter2D(img_np, -1, kernel)

cv.imshow("3x3 High Pass Filter MD 5", highpass_3x3)

# ----------------------------Low Pass Filter-------#

kernel_1 = np.array([[0, 1/8, 0],

                   [1/8, 1/2, 1/8],

                   [0, 1/8, 0]])



kernel_1 = (1 / 5) * (kernel_1)



lowpass_3x3 = cv.filter2D(img_np, -1, kernel_1)



cv.imshow("Simple 3x3 Low Pass 15 Filter", lowpass_3x3)

cv.waitKey(0)

cv.destroyAllWindows()



###############################################################################################

#laplacian

import cv2

import numpy as np

from scipy import ndimage



img = cv2.imread('/Users/

/Documents/DIP LAB/LAB 7/underexposed.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gaussian = cv2.GaussianBlur(gray,(3,3),0)



#canny

img_canny = cv2.Canny(img,100,200)



#sobel

img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)

img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)

img_sobel = img_sobelx + img_sobely





#prewitt

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)

img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)



#Laplacian

laplacian = cv2.Laplacian(gray,cv2.CV_64F)



#Robert

img = cv2.imread("/Users/

/Documents/DIP LAB/LAB 7/underexposed.jpg",0).astype('float64')

roberts_cross_v = np.array( [[1, 0 ],

                           [0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 1 ],

                           [ -1, 0 ]] )

img/=255.0

vertical = ndimage.convolve( img, roberts_cross_v )

horizontal = ndimage.convolve( img, roberts_cross_h )

edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))

edged_img*=255





cv2.imshow("Original Image", img)

cv2.imshow("Laplacian",laplacian)

cv2.imshow("Canny", img_canny)

cv2.imshow("Sobel X", img_sobelx)

cv2.imshow("Sobel Y", img_sobely)

cv2.imshow("Sobel", img_sobel)

cv2.imshow("Prewitt X", img_prewittx)

cv2.imshow("Prewitt Y", img_prewitty)

cv2.imshow("Prewitt", img_prewittx + img_prewitty)

cv2.imshow("Robert Edge",edged_img)





cv2.waitKey(0)

cv2.destroyAllWindows()





###############################################################################################

#houghman circle

import cv2

import numpy as np





img = cv2.imread('images.png', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



gray_blurred = cv2.blur(gray, (3, 3))



detected_circles = cv2.HoughCircles(gray_blurred,

              cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,

          param2 = 30, minRadius = 1, maxRadius = 40)

if detected_circles is not None:

  detected_circles = np.uint16(np.around(detected_circles))



  for pt in detected_circles[0, :]:

      a, b, r = pt[0], pt[1], pt[2]

      cv2.circle(img, (a, b), r, (0, 255, 0), 2)

      cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

      cv2.imshow("Detected Circle", img)

      cv2.waitKey(0)



###############################################################################################

# houghman line

import cv2

import numpy as np



# Read image

image = cv2.imread('underexposed.jpg')



# Convert image to grayscale

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



# Use canny edge detection

edges = cv2.Canny(gray,50,150,apertureSize=3)



# Apply HoughLinesP method to

# to directly obtain line end points

lines_list =[]

lines = cv2.HoughLinesP(

          edges, # Input edge image

          1, # Distance resolution in pixels

          np.pi/180, # Angle resolution in radians

          threshold=100, # Min number of votes for valid line

          minLineLength=5, # Min allowed length of line

          maxLineGap=10 # Max allowed gap between line for joining them

          )



# Iterate over points

for points in lines:

  # Extracted points nested in the list

  x1,y1,x2,y2=points[0]

  # Draw the lines joing the points

  # On the original image

  cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

  # Maintain a simples lookup list for points

  lines_list.append([(x1,y1),(x2,y2)])

 # Save the result image

cv2.imwrite('detectedLines.png',image)














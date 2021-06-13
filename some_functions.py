import numpy as np


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  channel_dict = {'L':1, 'RGB':3} # 'L' for Grayscale, 'RGB' : for 3 channel images
  return np.array(image.getdata()).reshape(
      (im_height, im_width, channel_dict[image.mode])).astype(np.uint8)

def zerolist(n):
    listofzeros = [0] * n
    return listofzeros


def kararver(karar):
    a = zerolist(7)
    if (karar[1*6]==1):
        a[0]=1
    else:
        a[0]=0
    if (karar[2*6]==1):
        a[1]=1
    else:
        a[1]=0
    if (karar[3*6]==1):
        a[2]=1
    else:
        a[2]=0
    if (karar[4*6]==1):
        a[3]=1
    else:
        a[3]=0
    if (karar[5*6]==1):
        a[4]=1
    else:
        a[4]=0
    if (karar[6*6]==1):
        a[5]=1
    else:
        a[5]=0
    if (karar[7*6]==1):
        a[6]=1
    else:
        a[6]=0
    return (a)
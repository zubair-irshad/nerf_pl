import numpy as np
import OpenEXR
import Imath

def EncodeToSRGB(v):
    return(np.where(v<=0.0031308,v * 12.92, 1.055*(v**(1.0/2.4)) - 0.055))

def exr2numpy(exr_path, chanel_name):
    '''
    See:
    https://excamera.com/articles/26/doc/intro.html
    http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    '''
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    
    channel_str = file.channel(chanel_name, Float_Type)
    
    channel = np.fromstring(channel_str, dtype = np.float32).reshape(size[1],-1)
    
    return(channel)

def load_image_from_exr(path):
    channels = []
    channel_names = ['R','G','B']
    for channel_name in channel_names:
        channel = exr2numpy(path, channel_name)
        channels.append(EncodeToSRGB(channel))
    RGB = np.dstack(channels)
    return RGB


def load_seg_from_exr(path):
    mask = exr2numpy(path, 'R')
    return mask

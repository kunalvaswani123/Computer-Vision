from matplotlib import pyplot as plt
import struct as st
import numpy as np

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

def data():
	filename = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
	labels = np.array([])
	data_types = {
	        0x08: ('ubyte', 'B', 1),
	        0x09: ('byte', 'b', 1),
	        0x0B: ('>i2', 'h', 2),
	        0x0C: ('>i4', 'i', 4),
	        0x0D: ('>f4', 'f', 4),
	        0x0E: ('>f8', 'd', 8)}
	for name in filename.keys():
		if name == 'images':
			imagesfile = open(filename[name],'rb')
		if name == 'labels':
			labelsfile = open(filename[name],'rb')
	imagesfile.seek(0)
	magic = st.unpack('>4B',imagesfile.read(4))
	if(magic[0] and magic[1]) or (magic[2] not in data_types):
		raise ValueError("File Format not correct")
	nDim = magic[3]
	#offset = 0004 for number of images
	#offset = 0008 for number of rows
	#offset = 0012 for number of columns
	#32-bit integer (32 bits = 4 bytes)
	imagesfile.seek(4)
	nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images/labels
	nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
	nC = st.unpack('>I',imagesfile.read(4))[0] #num of columns
	nBytes = nImg*nR*nC
	labelsfile.seek(8) #Since no. of items = no. of images and is already read
	#Read all data bytes at once and then reshape
	images = np.zeros((nImg,nR,nC))
	images = 255 - np.asarray(st.unpack('>'+'B'*nBytes,imagesfile.read(nBytes))).reshape((nImg,nR,nC))
	labels = np.asarray(st.unpack('>'+'B'*nImg,labelsfile.read(nImg))).reshape((nImg,1))
	images = np.array(images)

data()
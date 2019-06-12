from matplotlib import pyplot as plt
import struct as st
import numpy as np
import math

class NN:

	def __init__(self):
		self.a1 = np.random.uniform(-2,2,(784,1))
		self.a2 = np.random.uniform(-2,2,(130,1))
		self.a3 = np.random.uniform(-2,2,(10,1))
		self.z2 = np.random.uniform(-2,2,(130,1))
		self.z3 = np.random.uniform(-2,2,(10,1))
		self.b2 = np.random.uniform(-2,2,(130,1))
		self.b3 = np.random.uniform(-2,2,(10,1))
		self.d2 = np.random.uniform(-2,2,(130,1))
		self.d3 = np.random.uniform(-2,2,(10,1))
		self.w21 = np.random.uniform(-2,2,(130,784))
		self.w32 = np.random.uniform(-2,2,(10,130))
		self.learning_rate = 0.01

	def sigmoid(self,x):
		try:
			ans = math.exp(-x)
		except:
			return 0        
		return 1 / (1 + ans)

	def sigmoiddo(self,x):
		try:
			ans = math.exp(-x)
		except:
			return 0
		return (ans) / ((1 + ans) * (1 + ans))

	def gen_image(self,arr):
		two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
		plt.imshow(two_d, interpolation='nearest')
		return plt

	def sigm(self,x):
		vf = np.vectorize(self.sigmoid)
		return np.array([vf(xi) for xi in x])

	def sigmd(self,x):
		vf = np.vectorize(self.sigmoiddo)
		return np.array([vf(xi) for xi in x])

	def feedforward(self):
		self.z2 = self.w21 @ self.a1 + self.b2
		self.a2 = self.sigm(self.z2)
		self.z3 = self.w32 @ self.a2 + self.b3
		self.a3 = self.sigm(self.z3)

	def backpropogate(self,y):
		de = (self.a3 - y)
		de2 = self.sigmd(self.z3)
		self.d3 = np.multiply(de, de2)
		self.d2 = (self.w32.transpose()) @ self.d3
		de3 = self.sigmd(self.z2)
		self.d2 = np.multiply(self.d2,de3)
		self.b2 = self.b2 - self.learning_rate * self.d2
		self.b3 = self.b3 - self.learning_rate * self.d3
		dw32 = np.array(self.d3 @ self.a2.transpose())
		dw21 = np.array(self.d2 @ self.a1.transpose())
		self.w32 = self.w32 - self.learning_rate * dw32
		self.w21 = self.w21 - self.learning_rate * dw21 

newNN = NN();

''' DATA '''
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
# print(np.shape(images))
''' DATA ''' 

# ''' DATA2 '''
# filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 't10k-labels.idx1-ubyte'}
# labels2 = np.array([])
# for name in filename.keys():
# 	if name == 'images':
# 		imagesfile = open(filename[name],'rb')
# 	if name == 'labels':
# 		labelsfile = open(filename[name],'rb')
# imagesfile.seek(0)
# magic = st.unpack('>4B',imagesfile.read(4))
# if(magic[0] and magic[1]) or (magic[2] not in data_types):
# 	raise ValueError("File Format not correct")
# nDim = magic[3]
# #offset = 0004 for number of images
# #offset = 0008 for number of rows
# #offset = 0012 for number of columns
# #32-bit integer (32 bits = 4 bytes)
# imagesfile.seek(4)
# nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images/labels
# nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
# nC = st.unpack('>I',imagesfile.read(4))[0] #num of columns
# nBytes = nImg*nR*nC
# labelsfile.seek(8) #Since no. of items = no. of images and is already read
# #Read all data bytes at once and then reshape
# images2 = np.zeros((nImg,nR,nC))
# images2 = 255 - np.asarray(st.unpack('>'+'B'*nBytes,imagesfile.read(nBytes))).reshape((nImg,nR,nC))
# labels2 = np.asarray(st.unpack('>'+'B'*nImg,labelsfile.read(nImg))).reshape((nImg,1))
# # print(np.shape(images))
# ''' DATA2 '''

for _ in range(500):
	acc = 0 
	for i in range(60000):

		print(i)
		tempimage = images[i].reshape(784,1)
		newNN.a1 = tempimage
		y = np.zeros((10,1))
		a = labels[i]
		y[a] = 1
		newNN.feedforward()
		newNN.backpropogate(y)
		ind = np.argmax(newNN.a3)
		if ind == a:
			acc = acc + 1
		print((acc * 100) / (i+1))


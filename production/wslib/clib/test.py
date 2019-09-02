import ctypes
import numpy as np
import cv2

img_file = './rsmall.png'
so_file = './libws_c.so'

def main():
	lib = ctypes.cdll.LoadLibrary(so_file)

	lib.testlib()

	
	print('Open file: {}'.format(img_file))
	image_gray = cv2.imread(img_file, 0)
	#image_color = cv2.imread(file, cv2.IMREAD_COLOR)

	if type(image_gray) == type(None):
	    print('Open file {} failed.'.format(img_file))
	    return
	
	"""
	image_gray = np.array([ [0, 0, 5, 0, 1, 0], 
		  				    [2, 1, 3, 0, 0, 2], 
		  					[1, 4, 0, 1, 0, 3], 
		  					[0, 2, 0, 1, 2, 4], 
		  					[0, 0, 1, 0, 4, 5], 
		  					[0, 0, 0, 0, 4, 3], ], dtype = np.uint8)
	"""
	(h, w) = image_gray.shape[:2]
	src = np.ctypeslib.as_ctypes(image_gray)
	dst = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)
	lib.getCoreImage(src, dst, h, w, 0)
	dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
	coreImage = np.ctypeslib.as_array(dst, shape = image_gray.shape)

	print(coreImage.dtype)

	"""
	#np.set_printoptions(precision=3, suppress=True, formatter='int')
	print('=== src image ===')
	print(image_gray)
	print('=== dst image ===')
	print(coreImage)
	"""
	
	images = np.hstack([image_gray, coreImage])

	cv2.namedWindow('Image', flags = cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('Image', 1800, 1000)
	cv2.imshow('Image', images)
	k = cv2.waitKey(0)

	cv2.destroyAllWindows()
	

if __name__ == '__main__':
	main()
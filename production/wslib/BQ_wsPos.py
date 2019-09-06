import numpy as numpy
import cv2
import ctypes
import time
import argparse

import clib.BQ_clib as clib 
import pylib.BQ_downwardWelding as dwelding

class BQ_WsPos():

	def __init__(self, 
				 bottom_thick = 100, noisy_pixels = 20, 
				 thres_drop = 100, thres_normal = 50, move_limit = 3, 
				 ):
		
		self.lib = clib.initCLib("./clib/libws_c.so")
		self.value_array = []
		self.drop_array = []
		self.bottom_thick = bottom_thick
		self.noisy_pixels = noisy_pixels
		self.thres_drop = thres_drop
		self.thres_normal = thres_normal
		self.move_limit = move_limit

	def testlib(self):
		self.lib.testlib()

	def img2baseline(self):
		# 图像灰度转换
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

		# 黑场检测和阈值切割
		mean = gray.mean()
		self.black_limit = (int)(mean * 5)
		if self.black_limit > 245:
			self.black_limit = 245
		if self.black_limit < 3:
			self.black_limit = 3

		self.coreline = self.lib.getLineImage(
									self.lib, 
									self.gray, 
									black_limit = self.black_limit, 
									correct_angle = False)
		
		self.gaps = self.lib.fillLineGaps(
									self.lib, 
									self.coreline, 
									start_pixel = 5)
		
		self.coreline = self.gaps + self.coreline

	def getBottomCenter():
		# 从baseline获取底部中点位置，底部中点高度和包围位置。
		b_center, b_level, bound = dwelding.getBottomCenter(
												self.lib, 
												self.coreline, 
												bottom_thick = self.bottom_thick, 
												noisy_pixels = self.noisy_pixels)
		
		# 将center的输出值进行normalize处理，消除尖峰噪音干扰。
		self.center, self.value_array, self.drop_array = dwelding.normalizeCenter(
												self.value_array, b_center, 
												thres_drop = self.thres_drop, 
												thres_normal = self.thres_normal,
												move_limit = self.move_limit, 
												dropped_array = self.dropped_array)
		

	def phaseImage(image):
		self.image = image 







"""
完成图像的增强处理，包括图像分色，阈值切割，灰度转换等。
image: 输入RGB彩色图像
return: 输出增强后的灰度图像
"""
def img2baseline(image, ROI=None):
	# 图像灰度转换
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# 黑场检测和阈值切割
	mean = gray.mean()
	black_limit = (int)(mean * 5)
	if black_limit > 245:
		black_limit = 245

	if black_limit < 3:
		black_limit = 3

	coreline = getLineImage(lib, filt, black_limit = black_limit, correct_angle = False)
	gaps = fillLineGaps(lib, coreline, start_pixel = 5)
	result = gaps + coreline

	b_center, b_level, bound = getBottomCenter(lib, result, bottom_thick = BOTTOM_THICK, noisy_pixels = NOISY_PIXELS)
	


	return result


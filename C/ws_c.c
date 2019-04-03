#include <stdio.h>
#include <stdlib.h>

int WIDTH = 6;
int HEIGHT = 6;
unsigned char IMAGE[] = { 0, 0, 3, 0, 1, 0, 
		  		2, 1, 5, 0, 0, 2, 
		  		1, 4, 0, 1, 0, 3, 
		  		0, 2, 0, 1, 2, 4, 
		  		0, 0, 1, 0, 4, 5, 
		  		0, 0, 0, 0, 4, 3 }; 

unsigned char CORELINE[] = { 0,   0, 0, 0,   0,   0, 
		  					 0,   0, 0, 0,   0,   0, 
		  					 255, 0, 0, 0,   0,   0, 
		  					 0,   0, 0, 0,   0,   0, 
		  					 0,   0, 0, 0,   0,   0, 
		  					 0,   0, 0, 255, 255, 0 }; 

/*
typedef struct image
{
	int width;
	int height;
	int imageBuffer; 
} Image; 
*/
int testlib()
{
	printf("Lib load OK. \n");
	return 0;
}

int getCorePoint(unsigned char* src, unsigned char* out_max, int* out_pos, int h, int w, int column, int begin, int end)
{
	unsigned char max = 0;
	int sum = 0;
	//int length = end - begin;
	int i;
	unsigned char temp;

	for (i=begin; i<end; i++)
	{
		//printf("MAX: %d, SUM: %d, POINT: %d. \n", max, sum, src[i*w + column]);
		temp = src[i*w + column]; //src[column][i];

		sum += temp;

		if (temp > max)
		{
			max = temp;
		}
	}
	//printf("Found Max = %d, Sum = %d. \n", max, sum);

	sum = sum / 2;

	while (begin < end)
	{
		sum -= src[begin*w + column]; //src[column][begin];
		if (sum > 0)
		{
			begin ++;
			continue;
		}
		else
		{
			break;
		}
	}

	*out_pos = begin;
	*out_max = max;

	return 0; 
}

int getCoreImage(unsigned char* src, unsigned char* dst, int h, int w, unsigned char black_limit)
{   
	int scan_pos = 0;
	int seg_pos = 0;
	int pos = 0;
	unsigned char max = 0;

	int i, j;

	for (i=0; i<h; i++)
	{
		for (j=0; j<w; j++)
		{
			dst[i*w + j] = 0;
		}
	}
	//printf("\n");

	for (i=0; i<w; i++)
	{
		//printf("Start the column: %u\n", i);
		scan_pos = 0;

		while (scan_pos < h)
		{
			if (src[scan_pos*w + i] > black_limit)
			{
				seg_pos = scan_pos;
				//printf("Found seg_pos begin: %u \n", src[seg_pos*w + i]);

				while (seg_pos < h)
				//for (seg_pos=scan_pos; seg_pos<h; seg_pos++)
				{
					if (src[seg_pos*w + i] > black_limit)
					{
						seg_pos++;
					}
					else
					{
						//printf("Found seg_pos end: %u \n", src[seg_pos*w + i]);
						break;
					}
				}

				//printf("Call getCorePoint: column = %u, scan_pos = %u, seg_pos = %u. \n", i, scan_pos, seg_pos);
				getCorePoint(src, &max, &pos, h, w, i, scan_pos, seg_pos);
				//printf("\rDOT: (%u, %u), value: %3u.", i, pos, max);
				dst[pos*w + i] = max;

				scan_pos = seg_pos;
			}
			else
			{
				scan_pos++;
			}
		}
	} 

	//printf("\n");

    return 0; 
}

int followCoreLine(unsigned char* src, unsigned char* dst, int h, int w, int ref_level_left, int ref_level_right, int min_gap, int black_limit)
{
	int core_pos = 0; 
	int min_dist = h;
	int pre_level = ref_level_left;
	int i, j, temp;

	int* index = (int*)malloc(sizeof(int)*w);
	for (i=0; i<h; i++)
	{
		index[i] = -1;
	}

	for (i=0; i<h; i++)
	{
		for (j=0; j<w; j++)
		{
			dst[i*w + j] = 0;
		}
	}
	//printf("\n");

	for (i=0; i<w; i++)
	{	
		core_pos = -1;
		min_dist = h; 

		for (j=0; j<h; j++)
		{
			if (src[j*w + i] > black_limit)
			{
				temp = j - pre_level; 
				if (temp < 0)
				{
					temp = -temp;
				}

				if (temp < min_dist)
				{
					min_dist = temp;
					core_pos = j;
				}
				//printf("DOT: (%u, %u), pre_level: %u, core_pos: %u, min_dist: %u. \n", i, j, pre_level, core_pos, min_dist);
			}
		}

		//printf("Found column: %u, pre_level: %u, core_pos: %u, min_dist: %u. \n", i, pre_level, core_pos, min_dist);

		//if (core_pos < h && min_dist < min_gap)
		if (core_pos < h && core_pos > 0)
		{	
			index[i] = core_pos;
			dst[core_pos*w + i] = 255;
			//dst[core_pos*w + i] = src[core_pos*w + i];
			pre_level = core_pos;
		}
	}

	core_pos = 0; 
	min_dist = h;
	pre_level = ref_level_right;

	for (i=w-1; i>=0; i--)
	{	
		core_pos = 0;
		min_dist = h; 

		for (j=0; j<h; j++)
		{
			if (src[j*w + i] > black_limit)
			{
				temp = j - pre_level; 
				if (temp < 0)
				{
					temp = -temp;
				}

				if (temp < min_dist)
				{
					min_dist = temp;
					core_pos = j;
				}
				//printf("DOT: (%u, %u), pre_level: %u, core_pos: %u, min_dist: %u. \n", i, j, pre_level, core_pos, min_dist);
			}
		}

		//printf("Found column: %u, pre_level: %u, core_pos: %u, min_dist: %u. \n", i, pre_level, core_pos, min_dist);

		//if (core_pos < h && min_dist < min_gap)
		if (core_pos < h && core_pos > 0)
		{
			int left_p, mid_p;

			if (index[i] > 0)
			{
				// Caculate the left point. 
				if (i != 0)
				{
					left_p = index[i-1];
				}
				else
				{
					left_p = ref_level_left;
				}

				// Right point is the pre_level. 
				// Caculate the mid point. 
				if (left_p > 0)
				{
					mid_p = (left_p + pre_level) / 2;

					// if the new found pos is more approach to the mid point than the previous value.  
					if ( abs(core_pos - mid_p) > abs(index[i] - mid_p))
					{
						continue;
					}
				}
			}
			
			if (src[index[i]*w + i] <= src[core_pos*w + i])
			{
				dst[index[i]*w + i] = 0;
				index[i] = core_pos;
				dst[core_pos*w + i] = 255;
				//dst[core_pos*w + i] = src[core_pos*w + i];
				pre_level = core_pos;
			}

		}
	}

	return 0;
}

int fillLineGaps(unsigned char* coreLine, unsigned char* output, int h, int w, int black_limit)
{
	int i, j, m, n;
	int line_start = -1; 
	int gap_start = -1;
	int pre_value = -1;
	int found = -1; 
	float step = 0;

	for (i=0; i<w*h; i++)
	{
		output[i] = 0;
	}

	i = 0;
	while (i<w)
	{
		j = 0;

		while (j<h)
		{
			//printf("i: %d, j: %d, line_start: %d, gap_start: %d\n", i, j, line_start, gap_start);
			if (coreLine[j*w + i] > black_limit) //有像素点
			{	
				if (gap_start > 0) //之前都是没有像素点 
				{
					step = (j - pre_value) / (i - gap_start);
					for (m=gap_start; m<i; m++)
					{
						//printf("Set output pixel: %d, %d\n", m, pre_value+(int)step);
						output[(pre_value + (int)step)*w + m] = 255;
						step = step + step;
					}
				}
				else //之前也有像素点
				{
					//pre_value = coreLine[j*w + i];
					pre_value = j;
					//printf("Set pre_value: %d, %d\n", i, pre_value);
				} 

				gap_start = -1; 
				line_start = i;
				break; 
			}
			else // 没有像素点
			{
				j++;
			}

			if (j == h) //这一列都没找到像素点
			{
				if (gap_start < 0)
					gap_start = i;

				line_start = -1;
				//printf("No pixel found in: %d, gap_start: %d\n", i, gap_start);
			}

		}
		i++;
	}

	return 0; 

}


int fillLineGaps2(unsigned char* coreLine, unsigned char* output, int h, int w, int black_limit)
{
	int i, j, m, pre_value;
	int pre_black, pre_write;

	pre_black = 0; 
	pre_write = 0;
	pre_value = -1;

	for (i=0; i<w; i++)
	{
		for (j=0; j<h; j++)
		{
			//如果有点
			if (coreLine[j*w+i] > black_limit)
			{
				if (pre_black)
				{
					pre_black = 1;
					pre_write = 0; 
					break;
				}
			}
		}
	}

	return 0;
}

//int getBevelTop(unsigned char* coreLine, float* slope, int h, int w, int* bevelLeft, int* bevelRight, int judgeLength)
int getBevelTop(unsigned char* coreLine, float* slope, int h, int w)
{
	int i, j; 
	float pre_value = 20.0;

	for (i=0; i<w; i++)
	{
		slope[i] = pre_value;

		for (j=h-1; j>=0; j--)
		{
			if (coreLine[j*w + i] > 0)
			{
				pre_value = (float)(h - j + 1) / (float)(i + 1);
				slope[i] = pre_value;
			}
		}
	}


	return 0;
}

int coreLine2Index(unsigned char* coreLine, int h, int w, int* index)
{
	int i, j, pre_value;

	for (i=0; i<w; i++)
	{
		index[i] = h;

		for (j=h-1; j>=0; j--)
		{
			if (coreLine[j*w + i] > 0)
			{
				index[i] = j;
			}
		}
	}

	pre_value = -1; 

	/*
	FILL THE CODE FOR MISSING POINT.
	for (i=0; i<w; i++)
	{
		if (index[i])
	}
	*/

	return 0;
}


int fill2ColorImage(unsigned char* color, unsigned char* coreLine, int h, int w, int black_limit, unsigned char r, unsigned char g, unsigned char b)
{
	int i, j, index;

	for (i=0; i<h; i++)
	{
		for (j=0; j<w; j++)
		{	
			index = i * w + j;
			if (coreLine[index] > black_limit)
			{
				color[index*3]     = b; 
				color[index*3 + 1] = g;
				color[index*3 + 2] = r;
			}

		}
	}

	return 0;
}

int main(int argc, char const *argv[])
{
	//int* image = (int*)malloc(sizeof(int)*16);
	unsigned char* image = (unsigned char*)IMAGE;
	unsigned char* coreLine = (unsigned char*)CORELINE;

	unsigned char* output1 =  (unsigned char*)malloc(sizeof(unsigned char)*HEIGHT*WIDTH);
	unsigned char* output2 =  (unsigned char*)malloc(sizeof(unsigned char)*HEIGHT*WIDTH);
	unsigned char* output3 =  (unsigned char*)malloc(sizeof(unsigned char)*HEIGHT*WIDTH);

	int i, j;

	for (i=0; i<HEIGHT; i++)
	{
		for (j=0; j<WIDTH; j++)
		{
			output1[i*WIDTH + j] = 0;
			output2[i*WIDTH + j] = 0;
			output3[i*WIDTH + j] = 0;
		}
	}

	/*
	getCoreImage(image, output1, HEIGHT, WIDTH, 0);
	followCoreLine(output1, output2, HEIGHT, WIDTH, 2, 2, 3, 0);

	printf("\n===Original Image: ===");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", image[i*WIDTH + j]);
		}
	}
	printf("\n");

	printf("\n===Output1 Image: ===");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", output1[i*WIDTH + j]);
		}
	}
	printf("\n");

	printf("\n===Output2 Image: ===");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", output2[i*WIDTH + j]);
		}
	}
	printf("\n");
	*/

	printf("\n===CORELINE Image: ===");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", CORELINE[i*WIDTH + j]);
		}
	}
	printf("\n");

	fillLineGaps(coreLine, output3, HEIGHT, WIDTH, 0);
	printf("\n===fillLineGaps Image: ===");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", output3[i*WIDTH + j]);
		}
	}
	printf("\n");

	return 0;
}


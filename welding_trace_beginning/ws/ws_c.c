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


typedef struct image
{
	int width;
	int height;
	int imageBuffer; 
} Image; 

int testlib()
{
	printf("Lib load OK. \n");
	return 0;
}

int getCorePoint(unsigned char* src, unsigned char* out_max, int* out_pos, int h, int w, int column, int begin, int end)
{
	unsigned char max = 0;
	int sum = 0;

	int i;
	unsigned char temp;

	for (i=begin; i<end; i++)

		temp = src[i*w + column]; 

		sum += temp;

		if (temp > max)
		{
			max = temp;
		}
	}


	sum = sum / 2;

	while (begin < end)
	{
		sum -= src[begin*w + column]; 
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
		scan_pos = 0;

		while (scan_pos < h)
		{
			if (src[scan_pos*w + i] > black_limit)
			{
				seg_pos = scan_pos;

				while (seg_pos < h)
				{
					if (src[seg_pos*w + i] > black_limit)
					{
						seg_pos++;
					}
					else
					{
						break;
					}
				}

				getCorePoint(src, &max, &pos, h, w, i, scan_pos, seg_pos);
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
			}
		}

		if (core_pos < h && core_pos > 0)
		{	
			index[i] = core_pos;
			dst[core_pos*w + i] = 255;
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
			}
		}

		if (core_pos < h && core_pos > 0)
		{
			int left_p, mid_p;

			if (index[i] > 0)
			{
				if (i != 0)
				{
					left_p = index[i-1];
				}
				else
				{
					left_p = ref_level_left;
				}

				if (left_p > 0)
				{
					mid_p = (left_p + pre_level) / 2;

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
				pre_level = core_pos;
			}

		}
	}

	return 0;
}

int findPixelInColomn(unsigned char* coreline, int h, int w, int column)
{
	int i; 
	
	for (i=0; i<h; i++)
	{
		if (coreline[i*w + column] > 0)
		{
			return i;
		}
	}

	return -1;
}

int fillLineBetweenTwoPixels(unsigned char* output, int h, int w, int x1, int y1, int x2, int y2)
{
	int i, j;

	float step = (y2 - y1) / (x2 - x1);
	float cur_step = step; 

	for (i=x1; i<x2; i++)
	{
		output[(y1 + (int)cur_step)*w + i] = 255;
		cur_step += step; 
	}

	return 0;
}

int fillLineGaps(unsigned char* coreline, unsigned char* output, int h, int w, int start_pixel)
{
	int i, j;
	int pre_value = 0; // (>0) = pre y of the white, (0) = black. 
	int gap_start = -1; // Start of the gap. 
	int x, y;
	int x1, y1, x2, y2; 
	int end = start_pixel;

	// Find the first white pixel. 
	j = 0; 
	while (j<w)
	{
		y = findPixelInColomn(coreline, h, w, j);
		//printf("Y for %d column is %d\n", j, y);

		if (y<0)
		{
			j++;
		}
		else
		{
			gap_start = y; 
			pre_value = y; 

			if (end > 0)
			{	
				end --; 
				j ++; 
				continue;
			}
			else
			{
				break; 
			}
			
		}
	}

	
	for (i=j+1; i<w; i++)
	{
		y = findPixelInColomn(coreline, h, w, i);
		
		if (y < 0) 
		{
			if (pre_value > 0) 
			{
				x1 = i; 
				y1 = gap_start;
			}
			else  
			{
				continue;
			}
		}
		else 
		{
			if (pre_value < 0) 
			{
				x2 = i; 
				y2 = y; 

				fillLineBetweenTwoPixels(output, h, w, x1, y1, x2, y2);
			}
			else
			{
				gap_start = y;
			}
			
			
		}

		pre_value = y;
	}

	return 0;
}

int fillLineGaps2(unsigned char* coreLine, unsigned char* output, int h, int w, int black_limit)
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
			if (coreLine[j*w + i] > black_limit) 
			{	
				if (gap_start > 0) 
				{
					step = (j - pre_value) / (i - gap_start);
					for (m=gap_start; m<i; m++)
					{
						output[(pre_value + (int)step)*w + m] = 255;
						step = step + step;
					}
				}
				else 
				{
					pre_value = j;
				} 

				gap_start = -1; 
				line_start = i;
				break; 
			}
			else 
			{
				j++;
			}

			if (j == h) 
			{
				if (gap_start < 0)
					gap_start = i;

				line_start = -1;
			}

		}
		i++;
	}

	return 0; 

}



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

	fillLineGaps(coreLine, output3, HEIGHT, WIDTH, 1);
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


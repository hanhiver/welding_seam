#include <stdio.h>
#include <stdlib.h>

int WIDTH = 6;
int HEIGHT = 6;
char IMAGE[] = { 0, 0, 3, 0, 1, 0, 
		  		2, 1, 5, 0, 0, 2, 
		  		1, 4, 0, 1, 0, 3, 
		  		0, 2, 0, 1, 2, 4, 
		  		0, 0, 1, 0, 4, 5, 
		  		0, 0, 0, 0, 4, 3 }; 


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

int followCoreLine(unsigned char* src, unsigned char* dst, int h, int w, int ref_level, int min_gap, int black_limit)
{
	int core_pos = 0; 
	int min_dist = h;
	int pre_level = ref_level;
	int i, j, temp;

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

		if (core_pos < h && min_dist < min_gap)
		{
			dst[core_pos*w + i] = src[core_pos*w + i];
			pre_level = core_pos;
		}
	}

	return 0;
}

int main(int argc, char const *argv[])
{
	//int* image = (int*)malloc(sizeof(int)*16);
	unsigned char* image = IMAGE;

	unsigned char* output1 =  (unsigned char*)malloc(sizeof(unsigned char)*HEIGHT*WIDTH);
	unsigned char* output2 =  (unsigned char*)malloc(sizeof(unsigned char)*HEIGHT*WIDTH);

	int i, j;

	for (i=0; i<HEIGHT; i++)
	{
		for (j=0; j<WIDTH; j++)
		{
			output1[i*WIDTH + j] = 0;
			output2[i*WIDTH + j] = 0;
		}
	}

	getCoreImage(image, output1, HEIGHT, WIDTH, 0);
	followCoreLine(output1, output2, HEIGHT, WIDTH, 2, 3, 0);

	printf("\n===Original Image: ===\n");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", image[i*WIDTH + j]);
		}
	}

	printf("\n===Output1 Image: ===\n");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", output1[i*WIDTH + j]);
		}
	}

	printf("\n===Output2 Image: ===\n");
	for (i=0; i<HEIGHT; i++)
	{
		printf("\n");

		for (j=0; j<WIDTH; j++)
		{
			printf("%u\t", output2[i*WIDTH + j]);
		}
	}

	printf("\n");

	return 0;
}

#include <stdio.h>
#include <stdlib.h>

/*
typedef struct image
{
	int width;
	int height;
	int imageBuffer; 
} Image; 
*/

int getCorePoint(int* src, int* dst, int h, int w, int column, int begin, int end)
{
	int max = 0;
	int sum = 0;
	int length = end - begin;
	int i;
	int temp;

	for (i=0; i<length; i++)
	{
		temp = src[column*w + i]; //src[column][i];

		sum += temp;

		if (temp > max)
		{
			max = temp;
		}
	}

	sum = sum / 2;

	while (begin < end)
	{
		sum -= src[column*w + begin]; //src[column][begin];
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

	dst[0] = begin;
	dst[1] = max;

	return 0; 
}

int getCoreImage(int* src, int* dst, int h, int w, int black_limit)
{   
	int scan_pos = 0;
	int seg_pos = 0;
	int ret[2];

	int i, j;
	int pos, value;

	for (i=0; i<w; i++)
	{
		printf("Start the column: %d\n", i);
		scan_pos = 0;

		while (scan_pos < h)
		{
			if (src[scan_pos*w + i] > black_limit)
			{
				for (seg_pos=scan_pos; seg_pos<h; seg_pos++)
				{
					if (src[seg_pos*w + i] <= black_limit)
					{
						printf("check: %d \n", src[seg_pos*w + i]);
						break;
					}

					//pos, value = getCorePoint(image[..., i], scan_pos, seg_pos)
					getCorePoint(src, ret, h, w, i, scan_pos, seg_pos);
					pos = ret[0];
					value = ret[1];
					printf("\rDOT: (%d, %d), value: %d.", i, pos, value);
					dst[pos*w + i] = value;

					scan_pos = seg_pos;
				}
			}
			else
			{
				scan_pos++;
			}
		}
	} 

    return 0; 
}

int main(int argc, char const *argv[])
{
	//int* image = (int*)malloc(sizeof(int)*16);
	int image[] = { 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0 }; 

	int* output =  (int*)malloc(sizeof(int)*16);

	getCoreImage(image, output, 4, 4, 0);

	for (int i=0; i<4; i++)
	{

		printf("\n");

		for (int j=0; j<4; j++)
		{
			printf("%d\t", output[i*4 + j]);
		}
	}

	return 0;
}


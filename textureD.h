#include <opencv2/core/core.hpp>

double calTtureDsity(const Mat& img, const int CLOW, const int CHIG, double area)
{
	double edge = 0;
	int width = img.cols;
	int heigh = img.rows;
	uchar *p = NULL;
	for (int i = 0; i < heigh; i++)
	{
		const uchar *p = img.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			if (int(p[j]) == 255)
				edge++;
		}
	}

	//±ßÔµÃÜ¶È¼ÆËã
	return edge / area;
}
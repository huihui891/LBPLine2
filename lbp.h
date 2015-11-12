//====================================================================  
// 作者   : quarryman  
// 邮箱   : quarrying{at}qq.com  
// 主页   : http://blog.csdn.net/quarryman  
// 日期   : 2013年08月11日  
// 描述   : Uniform Pattern的LBP  
//==================================================================== 
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <math.h>
/*#include <atltrace.h>*/


/*#pragma comment(lib,"atlsd.lib")*/

using namespace std;
using namespace cv;

static int getHopCount(uchar i)
{
	int a[8]={0};
	int k=7;
	int cnt=0;
	while(i)
	{
		a[k]=i&1;
		i>>=1;
		--k;
	}
	for(int k=0;k<8;++k)
	{
		if(a[k]!=a[k+1==8?0:k+1])
		{
			++cnt;
		}
	}
	return cnt;
}

static void lbp59table(uchar* table)
{
	memset(table,0,256);
	uchar temp=1;
	for(int i=0;i<256;++i)
	{
		if(getHopCount(i)<=2)
		{
			table[i]=temp;
			temp++;
		}
		// printf("%d\n",table[i]);
	}
}

static void LBP(IplImage* src, IplImage* dst)
{
	int width=src->width;
	int height=src->height;
	uchar table[256];
	lbp59table(table);
	for(int j=1;j<width-1;j++)
	{
		for(int i=1;i<height-1;i++)
		{
			uchar neighborhood[8]={0};
			neighborhood[7]	= CV_IMAGE_ELEM( src, uchar, i-1, j-1);
			neighborhood[6]	= CV_IMAGE_ELEM( src, uchar, i-1, j);
			neighborhood[5]	= CV_IMAGE_ELEM( src, uchar, i-1, j+1);
			neighborhood[4]	= CV_IMAGE_ELEM( src, uchar, i, j+1);
			neighborhood[3]	= CV_IMAGE_ELEM( src, uchar, i+1, j+1);
			neighborhood[2]	= CV_IMAGE_ELEM( src, uchar, i+1, j);
			neighborhood[1]	= CV_IMAGE_ELEM( src, uchar, i+1, j-1);
			neighborhood[0]	= CV_IMAGE_ELEM( src, uchar, i, j-1);
			uchar center = CV_IMAGE_ELEM( src, uchar, i, j);
			uchar temp=0;

			for(int k=0;k<8;k++)
			{
				temp+=(neighborhood[k]>=center)<<k;
			}
			//CV_IMAGE_ELEM( dst, uchar, i, j)=temp;
			CV_IMAGE_ELEM( dst, uchar, i, j)=table[temp];
		}
	}
}

static bool ulbpIndex(vector<int> &uniform_lbp)
{
	uniform_lbp.clear();
	for (int i = 0; i < 256; ++i)
	{
		int data = i;
		int dataTmp1 = data, dataTmp2 = data;
		int jump = 0; int tmp = -1;


		for (int k = 0; k < 8; ++k)
		{
			dataTmp1 = data >> 1;
			dataTmp2 = dataTmp1 << 1;
			int t = data ^ dataTmp2;
			data = dataTmp1;
			assert(t == 0 || t == 1);
			if (tmp == -1 && k == 0)
			{
				tmp = t;
				continue;
			}
			if (tmp != t)
			{
				jump++;
				tmp = t;
			}
		}


		if (jump < 3)
		{
			uniform_lbp.push_back(i);
		}
	}
	if (uniform_lbp.size() != 58)
	{
		return false;
	}
	return true;

}

//------------------------------------------------------------------------------
// LBPH
//------------------------------------------------------------------------------

template <typename _Tp> static
	void olbp_(InputArray _src, OutputArray _dst) {
		// get matrices
		Mat src = _src.getMat();
		// allocate memory for result
		_dst.create(src.rows-2, src.cols-2, CV_8UC1);
		Mat dst = _dst.getMat();
		// zero the result matrix
		dst.setTo(0);

		//uniform lbp
		uchar table[256];
		lbp59table(table);

		// calculate patterns
		for(int i=1;i<src.rows-1;i++) {
			for(int j=1;j<src.cols-1;j++) {
				_Tp center = src.at<_Tp>(i,j);
				unsigned char code = 0;
				code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
				code |= (src.at<_Tp>(i-1,j) >= center) << 6;
				code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
				code |= (src.at<_Tp>(i,j+1) >= center) << 4;
				code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
				code |= (src.at<_Tp>(i+1,j) >= center) << 2;
				code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
				code |= (src.at<_Tp>(i,j-1) >= center) << 0;
				//uniform lbp
				dst.at<unsigned char>(i-1,j-1) = table[code];
			}
		}
}


//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
	inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
		//get matrices
		Mat src = _src.getMat();
		// allocate memory for result
		_dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
		Mat dst = _dst.getMat();
		// zero
		dst.setTo(0);

		//uniform lbp
// 		uchar table[256];
// 		lbp59table(table
		vector<int> table;
		table.reserve(100);
		ulbpIndex(table);


		for(int n=0; n<neighbors; n++) {
			// sample points
			float x = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
			float y = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
			// relative indices
			int fx = static_cast<int>(floor(x));
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));
			int cy = static_cast<int>(ceil(y));
			// fractional part
			float ty = y - fy;
			float tx = x - fx;
			// set interpolation weights
			float w1 = (1 - tx) * (1 - ty);
			float w2 =      tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 =      tx  *      ty;

			int index = 0;
			// iterate through your data
			for(int i=radius; i < src.rows-radius;i++) {
				for(int j=radius;j < src.cols-radius;j++) {
					// calculate interpolated value
					float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
					// floating point precision, so check some machine-dependent epsilon 
					// uniform lbp
					dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || 
						(std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) 
						<< n;
				}
			}
		}

		//printMat<int>(dst);

		for (int i = 0; i < dst.rows; ++i)
		{
			for (int j = 0; j < dst.cols; ++j)
			{
				int data = dst.at<int>(i, j);
				vector<int>::iterator iter = find(table.begin(), table.end(), data);
				if (iter == table.end())
				{
					dst.at<int>(i, j) = 0;
				}
				else
				{
					int new_data = iter - table.begin() ;
					dst.at<int>(i, j) = new_data + 1;
				}
			}
		}

		//printMat<int>(dst);
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
	int type = src.type();
	switch (type) {
	case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
	case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
	case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
	case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
	case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
	case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
	case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
	default:
		string error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
		CV_Error(CV_StsNotImplemented, error_msg);
		break;
	}
}

static Mat
	histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false)
{
	Mat result;
	// Establish the number of bins.
	int histSize = maxVal-minVal+1;
	// Set the ranges.
	float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
	const float* histRange = { range };
	// calc histogram
	calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
	// normalize
	if(normed) {
		result /= (int)src.total();
	}
	return result.reshape(1,1);
}

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
	Mat src = _src.getMat();
	switch (src.type()) {
	case CV_8SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_8UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_16SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_16UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_32SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_32FC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	default:
		CV_Error(CV_StsUnmatchedFormats, "This type is not implemented yet."); break;
	}
	return Mat();
}

static Mat spatial_histogram(InputArray _src, int numPatterns,
	int grid_x, int grid_y, bool /*normed*/)
{
	Mat src = _src.getMat();
	// calculate LBP patch size
	int width = src.cols/grid_x;
	int height = src.rows/grid_y;
	// allocate memory for the spatial histogram
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given
	if(src.empty())
		return result.reshape(1,1);
	// initial result_row
	int resultRowIdx = 0;
	// iterate through grid
	for(int i = 0; i < grid_y; i++) {
		for(int j = 0; j < grid_x; j++) {
			Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
			Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);
			// copy to the result matrix
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1,1);
}

//------------------------------------------------------------------------------
// wrapper to cv::elbp (extended local binary patterns)
//------------------------------------------------------------------------------

static Mat elbp(InputArray src, int radius, int neighbors) {
	Mat dst;
	elbp(src, dst, radius, neighbors);
	return dst;
}


template <typename _Tp> static
void printMat(Mat& img)
{
	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < height; i++)
	{
		//uchar *p = img.ptr<_Tp>(i);
		for (int j = 0; j < width; j++)
		{
			cout << double(img.at<_Tp>(i,j)) << " ";
		}
		cout << endl;
	}
}

template <typename _Tp> static
void printMat2File(Mat& img, char *f)
{
	ofstream fout(f, ios::out | ios::app);
	if (!fout.is_open())
	{
		cerr << "Can't open " << f << "file for output!\n";
		exit(EXIT_FAILURE);
	}

	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < height; i++)
	{
		const _Tp *p = img.ptr<_Tp>(i);
		for (int j = 0; j < width; j++)
		{
			fout << (p[j]) << " ";
		}
		fout << "\n";
	}

	fout.close();
}

template <typename _Tp> static
void printIpl(IplImage* img)
{
	int height = img->height;
	int width = img->width;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << int(CV_IMAGE_ELEM(img,_Tp,i,j)) << " ";
		}
		cout << endl;
	}
}


template <typename _Tp> static
void energy(const Mat& hist, double* e)
{
	assert(hist.rows == 1);

	double e1,e2;
	e1 = e2 = 0.0;
	int H = hist.cols;

	for (int i = 0; i < hist.rows; i++)
	{
		const _Tp *p = hist.ptr<_Tp>(i);
		for (int j = 0; j < H; j++)
		{
			if (p[j] < 1.0)
				continue;
			else
			{
				e1 = e1 +  (p[j])*log((p[j]));
				e2 = e2 +  float(p[j])*(p[j])*log((p[j]))*log((p[j]));
			}
		}
	}

	e1 = -1 * e1 / H;
	e2 = -1 * e2 / (H*H);

	e[0] = e1;
	e[1] = e2;
}



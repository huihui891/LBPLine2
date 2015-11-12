
#include "lbp.h"
#include "fcm.h"
#include "textureD.h"
#include <vector>

const int WSize = 24;
const int WStep = 6;

const int CLOW = 100;
const int CHIG = 250;

const int radius = 2;
const int samplePoints = 8;

//����ǿ����������

const int num_dim         = 3;
const int num_cluster     = 2;
const double epsilon      = 0.05;
const double fuzziness    = 2;



int main()
{
	double start = double(getTickCount());

	//1 ����LBPͼ
	Mat img = imread("ori.jpg",0);
	Mat imgLBP = elbp(img,radius,samplePoints);
	//imwrite("imgLBP.jpg",imgLBP);
	//printMat<int>(imgLBP);
	//printMat2File<int>(imgLBP,"lbp.txt");

	//2 ����Cannyͼ
	Mat imgCy;
	Mat RoI = img(Rect(radius,radius,img.cols-2*radius,img.rows-2*radius));
	Canny(RoI,imgCy,CLOW,CHIG);
	//imwrite("canny.jpg",imgCy);

	//���Сͼ��
	const int heigh = imgLBP.rows;
	const int width = imgLBP.cols;
	const int num_points = ((width-WSize)/WStep + 1)*((heigh-WSize)/WStep + 1);
	double area  = WSize * WSize;
	int index = 0;
	vector<vector<double>> dataBase(num_points);
	for (int i = 0; i < num_points; i++)
		dataBase[i].resize(3);
	for (int i = 0; i < heigh - WSize; i += WStep)
	{
		for (int j = 0; j < width - WSize; j += WStep)
		{
			Rect r(j,i,WSize,WSize);
			Mat subLBP = imgLBP(r);
			Mat subCy  = imgCy(r);

			//3 ����ֱ��ͼ������
			Mat hist = histc(subLBP,0,58,false);
			//printMat<float>(hist);
			double e[2] = {0.0};
			energy<float>(hist,e);

			//4 ���������ܶ�
			//printMat<uchar>(subCy);
			double ratio =  calTtureDsity(subCy,CLOW,CHIG,area);

			//5 ������������
			dataBase[index][0] = e[0];
			dataBase[index][1] = e[1];
			dataBase[index][2] = ratio;
		
			index++;
		}
	}


	//6 FCM������
 	CFCM f(num_points,num_dim,num_cluster,epsilon,fuzziness);
 	f.fcm(dataBase);
/* 	//��ӡ������ֵ
	char *of = "lishudu.txt";
	ofstream fout(of, ios::out | ios::app);
	if (!fout.is_open())
	{
		cerr << "Can't open " << of << "file for output!\n";
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < num_points; i++)
	{
		for (int j = 0; j < num_cluster; j++)
		{
			fout << f.degree_of_memb[i][j] << " ";
		}
		fout << endl;
	}
	*/


	//7 ������������ֵ ��0.5Ϊ��׼
	const int pitchsOfEachRow = (width-WSize)/WStep + 1;
	//const int pitchsOfEachCol = ceilf((heigh-WSize)/WStep);
	Mat segImg(imgLBP.size(),CV_8UC1);
	segImg.setTo(0);
	for (int i = 0; i < num_points; i++)
	{
		if (f.degree_of_memb[i][0] > 0.5)
		{
			//�����ǵڶ���
			//��������ͼ���е�λ�� ���ﻹ����Щ�����
			int m = i / pitchsOfEachRow * WStep;
			int n = i % pitchsOfEachRow * WStep;
			assert(m <= heigh - WSize);
			assert(n <= width - WSize);
			for (int p = m; p < m + WSize; p++)
			{
				uchar *k = segImg.ptr<uchar>(p);
				for (int q = n; q < n + WSize; q++)
					k[q] = 255;
			}
		}
	}

	double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	cout << "It took " << duration_ms << " ms." << endl;
	imshow("seg",segImg);

	cvWaitKey(0);

	return 0;
}


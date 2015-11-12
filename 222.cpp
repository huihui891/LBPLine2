
bool ulbpIndex(vector<int> &uniform_lbp)
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

// ULBP特征提取

template <typename _Tp> static
        inline void ulbp_(InputArray _src, OutputArray _dst, int radius, int neighbors, vector<int> m_uniform) 
    {
        if (neighbors != 8 || m_uniform.size() != 58)
        {
            cout << "neighbor must be 8! and uniform size be 58!\n";
            system("pause");
            exit(-1);
        }


        //get matrices
        Mat src = _src.getMat();
        // allocate memory for result
        _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
        Mat dst = _dst.getMat();
        // zero
        dst.setTo(0);
        for(int n=0; n<neighbors; n++) 
        {
            // sample points
            float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
            float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
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
            // iterate through your data
            for(int i=radius; i < src.rows-radius;i++) {
                for(int j=radius;j < src.cols-radius;j++) {
                    // calculate interpolated value
                    float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                    // floating point precision, so check some machine-dependent epsilon
                    dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || 
                        (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) 
                        << n;
                }
            }
        }


        for (int i = 0; i < dst.rows; ++i)
        {
            for (int j = 0; j < dst.cols; ++j)
            {
                int data = dst.at<int>(i, j);
                vector<int>::iterator iter = find(m_uniform.begin(), m_uniform.end(), data);
                if (iter == m_uniform.end())
                {
                    dst.at<int>(i, j) = 0;
                }
                else
                {
                    int new_data = iter - m_uniform.begin() ;
                    dst.at<int>(i, j) = new_data + 1;
                }
            }
        }
    }
}


// ULBP特征映射表
int mapping[256] = 
{
    0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 
    11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 
    16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
    17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 
    22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
    23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
    24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 
    29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 
    36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 
    42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 
    47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57
};


//Compute an single lbp code from a pixel
int LBPFeatureExtractor::lbpCode (unsigned char seq[9])
{
    bool lab[8] = {false};
    int base = seq[0];
    int result = 0, one = 1, final;
    int i;
    for (i = 0; i < 8; i++) {
        if (base >= seq[i+1])
            lab[i] = 1;
        else
            lab[i] = 0;
    }


    for (i = 0; i < 8; i++, one = one << 1)
        result += lab[i]*one;


    if (result >= 256)
    {
        printf("LBP编码错误\n");
        system("pause");
        exit(-1);
    }
    return mapping[result];
}


//Compute lbp histgoram from an image
int* lbpHist(Mat &img, int* lbpHist)
{
    unsigned char locP[9];
    for(int i=0; i<59; i++)
        lbpHist[i] = 0;
    for(int i = 1; i<img.rows-1; i++)
        for(int j = 1; j< img.cols-1; j++)
        {  
            locP[0] = img.at<unsigned char>(i,j);
            locP[1] = img.at<unsigned char>(i-1,j);
            locP[2] = img.at<unsigned char>(i-1,j-1);
            locP[3] = img.at<unsigned char>(i,j-1);
            locP[4] = img.at<unsigned char>(i+1,j-1);
            locP[5] = img.at<unsigned char>(i+1,j);
            locP[6] = img.at<unsigned char>(i+1,j+1);
            locP[7] = img.at<unsigned char>(i,j+1);
            locP[8] = img.at<unsigned char>(i-1,j+1);
            lbpHist[lbpCode(locP)]++;
        }
    return lbpHist;
}





int* extractAt(const Mat &inputImage,  const vector< pair<double, double> >points,   int *outputFeature){


    Mat processedImage;
    Mat resizeImage;
    //convert image into graylevel
    // 将彩色图像转换为灰度图像
    if (inputImage.channels() == 3) {
        cvtColor(inputImage, processedImage, CV_BGR2GRAY );
    } else {
        processedImage = inputImage;
    }
    equalizeHist(processedImage, processedImage);
    // 对于每个尺度下的图像提取LBP特征
    int count = 0;
    for(int s = 0; s<this->scale.size(); s++)
    {
        int currentScale = this->scale[s];
        //scale face into proper size
        // 将关键点映射到当前尺度
        vector< pair<double, double> > newPoints;
        for(int i=0; i<points.size(); i++)
        {
            pair<double,double> point;
            point.first = points[i].first*double(currentScale)/inputImage.cols;
            point.second = points[i].second*double(currentScale)/inputImage.rows;
            newPoints.push_back(point);
        }


        //printf("%d %d %d\n", currentScale, inputImage.cols, inputImage.rows);
        // 直方图均衡化后缩放图像，将直方图均衡化放到缩放尺度之外，没有必要每次都做
        // equalizeHist(processedImage, processedImage);
        resize(processedImage, resizeImage, Size(currentScale, currentScale));
        //compute center and extract lbp feature
        Mat patch;
        int hist[59];
        // 对于每个关键点提取，patchsize=10，numCellX=4,numCellY=4
        for(int i=0; i<newPoints.size(); i++)
        {
            for(int j=0; j<numCellX; j++)
                for(int k=0; k<numCellY; k++)
                {
                    double centerX = int(newPoints[i].first - patchSize*(numCellX/2) + (numCellX%2 == 0)*patchSize*0.5 + patchSize*j);
                    double centerY = int(newPoints[i].second - patchSize*(numCellY/2) + (numCellY%2 == 0)*patchSize*0.5 + patchSize*k);
                    //circle(resizeImage, Point(centerX,centerY), 5, Scalar(255,0,0), CV_FILLED);
                    getRectSubPix(resizeImage, Size(patchSize+2, patchSize+2), Point2f(centerX,centerY), patch);
                    lbpHist(patch, hist);
                    for(int l=0; l<59; l++)
                        outputFeature[count++] = hist[l];
                }
        }
        //imwrite(to_string(currentScale)+"out.jpg", resizeImage);
        resizeImage.release();
    }
    processedImage.release();
    return NULL;
}

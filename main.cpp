#include<eigen3/Eigen/Dense>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/opencv.hpp>
#include<array>
#include<iostream>
#include<vector>
#include<math.h>
#include<string>
#include<thread>
#include<atomic>
#include"/home/jiing/zhuangjiaban/include/numeric_rm.h"
#include"/home/jiing/zhuangjiaban/include/opencv_extended.h"
#include "/home/jiing/zhuangjiaban/include/serial.h"
#include"/home/jiing/zhuangjiaban/include/predictor.h"
#include"/home/jiing/zhuangjiaban/include/armordector.h"
#include"/home/jiing/zhuangjiaban/include/anglesolver.h"


using namespace std;
using namespace cv;

#define DEBUG_DETECTION
//#define GET_ARMOR_PICTURE

#define MOUSE_DEBUG 0//开启鼠标动态调参功能

#define BUFFER_SIZE 1
#define BUFFER_SIZE_UART 1

#define PREDICT 1

volatile unsigned int prdIdx;
volatile unsigned int csmIdx;


volatile unsigned int prdIdx_uart;
volatile unsigned int csmIdx_uart;


int _trackCnt = 0;
int _allCnt = 0;

Mat srcImg;
Mat src;
Rect _roi;



vector<Mat> channels;
Mat img_gray, img_hsv;
Mat binbrightimg;
bool stop = false;
ArmorParam _param;//定义装甲板匹配参数
ArmorDescriptor _targetArmor;//目标装甲板
std::vector<cv::Point2f> armorVetex;//装甲板四个角点坐标
AngleSolver _solverPtr;
AngleSolverParam angleParam;
Vec2f targetAngle;
double distance_armor;

const int statenum = 4;
const int measurenmu = 2;
KalmanFilter KF(statenum, measurenmu, 0);
Mat state(statenum, 1, CV_32FC1);
Mat processnoise(statenum, 1, CV_32F);
Mat measurement = Mat::zeros(measurenmu,1,CV_32F);

/*
        图像预处理使用的Mat变量
*/
cv::Mat color_light;
std::vector<cv::Mat> bgr_channel;

cv::Mat binary_brightness_img; // 亮度二值化
cv::Mat binary_color_img;      // 颜色二值化

/*
 *串口通信变量
*/
Uart Serial;
std::atomic<bool> InitUartAccomplish(false);//串口通信初始化成功标志

Predictor predictor;

typedef enum 
{
	DETECTING_STATE, TRACKING_STATE
} Status;

Status status;
int armor_detect_flag=0;//装甲板检测状态

struct ImageData {
	Mat img;
	unsigned int frame;
};
ImageData imagedata[BUFFER_SIZE];


struct Uartdata{
  float yaw;
  float pitch;
  float distance;

};

Uartdata uartdata[BUFFER_SIZE_UART];


/*
        矫正灯条
*/

cv::RotatedRect& adjustRec(cv::RotatedRect& rec, const int mode)
{
        using std::swap;

        float& width = rec.size.width;
        float& height = rec.size.height;
        float& angle = rec.angle;

        if (mode == 1)
        {
                if (width < height)
                {
                        swap(width, height);
                        angle += 90.0;
                }
        }

        while (angle >= 90.0) angle -= 180.0;
        while (angle < -90.0) angle += 180.0;

        if (mode == 2)
        {
                if (angle >= 45.0)
                {
                        swap(width, height);
                        angle -= 90.0;
                }
                else if (angle < -45.0)
                {
                        swap(width, height);
                        angle += 90.0;
                }
        }

        return rec;
}


void klamanfilter_init()
{

	//randn(state, Scalar::all(0), Scalar::all(0.1));
	KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
	setIdentity(KF.measurementMatrix, Scalar::all(1));
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//原为1e-1
	setIdentity(KF.errorCovPost, Scalar::all(1));
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
}
cv::Point klamanfilter_point(const cv::Point2f point)
{
	double center_x = point.x;
	double center_y = point.y;

	Mat prediction = KF.predict();
	Point predictPt = Point(prediction.at<float>(0), prediction.at<float>(1));

	measurement.at<float>(0) = (float)center_x;
	measurement.at<float>(1) = (float)center_y;

	Mat kalman_caulc=KF.correct(measurement);

	center_x = prediction.at<float>(0);
	center_y = prediction.at<float>(1);

	Point kalman_point;
		//kalman_point = predictPt;
	kalman_point.x = kalman_caulc.at<float>(0);
	kalman_point.y = kalman_caulc.at<float>(1);

	return kalman_point;
}

int brightness_threshold, color_diff;
int EXPOSURE;
Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
/*
	动态调节HSV阈值参数函数
*/
void trackbar(int, void*)
{
	cv::threshold(img_gray, binary_brightness_img, brightness_threshold, 255, CV_THRESH_BINARY);
	imshow("binary_light", binary_brightness_img);
	cv::threshold(color_light, binary_color_img, color_diff, 255, CV_THRESH_BINARY);
	cv::dilate(binary_color_img, binary_color_img, element);
	imshow("binary_color", binary_color_img);
	binbrightimg = binary_color_img & binary_brightness_img;
	cv::dilate(binbrightimg, binbrightimg, element);
	imshow("yuchuli", binbrightimg);
	//cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0.25);
	//cap.set(CV_CAP_PROP_EXPOSURE, 0.8);
	//std::cout<<cap.get(CV_CAP_PROP_EXPOSURE)<<endl;
	//imshow("show", src);
	cv::waitKey(5);

}

int detect( cv::Mat& frame )
{
	//cv::resize(frame, frame, cv::Size(1280, 720));
	//frame = frame(Rect(ROI_START_X,ROI_START_Y,1280-ROI_START_X,720-ROI_START_Y));//若划定ROI ,则对_targetArmor.center坐标的X，Y值加上对应ROI的起始坐标X,Y值
	//Rect _roi = Rect(Point(0, 0), frame.size());
	std::vector<LightDescriptor>lightInfors;
	std::vector<ArmorDescriptor> _armors;

	cv::split(frame, bgr_channel);//分离图像通道
	/*
		检测灯条
	*/
	{
		_armors.clear();

		
		if (_param.enemy_color == BLUE)
		{
                        cv::subtract(bgr_channel[0], bgr_channel[2], color_light);
			cv::cvtColor(frame, img_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
			cv::threshold(img_gray, binary_brightness_img, _param.brightness_threshold_blue, 255, CV_THRESH_BINARY);
			//imshow("binary_light", binary_brightness_img);
			cv::threshold(color_light, binary_color_img, _param.blue_color_diff, 255, CV_THRESH_BINARY);
			cv::dilate(binary_color_img, binary_color_img, element);
			//imshow("binary_color", binary_color_img);
			binbrightimg = binary_color_img & binary_brightness_img;
			cv::dilate(binbrightimg, binbrightimg, element);
			//imshow("yuchuli", binbrightimg);
		}

		else if(_param.enemy_color == RED)
		{
                        cv::subtract(bgr_channel[2], bgr_channel[0], color_light);
			cv::cvtColor(frame, img_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
			cv::threshold(img_gray, binary_brightness_img, _param.brightness_threshold_red, 255, CV_THRESH_BINARY);
			//imshow("binary_light", binary_brightness_img);
			cv::threshold(color_light, binary_color_img, _param.red_color_diff, 255, CV_THRESH_BINARY);
			cv::dilate(binary_color_img, binary_color_img, element);
			//imshow("binary_color", binary_color_img);
			binbrightimg = binary_color_img & binary_brightness_img;
			cv::dilate(binbrightimg, binbrightimg, element);
			//imshow("yuchuli", binbrightimg);
		}
		
		vector<vector<Point>> lightcontours;
		findContours(binbrightimg.clone(), lightcontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		for (const auto& contour : lightcontours)
		{
			float lightcontourarea = contourArea(contour);//计算轮廓面积
			if (contour.size() <= 5 || lightcontourarea < _param.light_min_area)//通过面积初步筛选
				continue;
			RotatedRect lightRec = fitEllipse(contour);
			adjustRec(lightRec, 2);//矫正灯条
			float ratio = lightRec.size.width / lightRec.size.height;//长比宽
			if (ratio > _param.light_max_ratio || ratio < _param.light_min_ratio || lightcontourarea / lightRec.size.area() < _param.light_contour_min_solidity)//矫正后筛选
				continue;

			lightRec.size.width *= _param.light_color_detect_extend_ratio;//扩大符合条件的灯条的尺寸1.2倍
			lightRec.size.height *= _param.light_color_detect_extend_ratio;
			Rect lightRect = lightRec.boundingRect();
			const Rect srcBound(Point(0, 0), frame.size());
			lightRect &= srcBound;
			Mat lightimg = frame(lightRect);
			Mat lightMask = Mat::zeros(lightRect.size(), CV_8UC1);
			Point2f lightVertexArray[4];
			lightRec.points(lightVertexArray);
			std::vector<Point> lightVertex;
			for (int i = 0; i < 4; i++)
			{
				lightVertex.emplace_back(Point(lightVertexArray[i].x - lightRect.tl().x,
					lightVertexArray[i].y - lightRect.tl().y));
			}
			fillConvexPoly(lightMask, lightVertex, 255);
			if (lightimg.size().area() <= 0 || lightMask.size().area() <= 0)//根据灯条包围矩阵面积筛选
				continue;
			cv::dilate(lightMask, lightMask, element);
			const Scalar meanVal = mean(lightimg, lightMask);
			lightInfors.push_back(LightDescriptor(lightRec));
			//std::cout << "there are all lights" << endl;
		}
#ifdef DEBUG_DETECTION
		vector<RotatedRect>lightsRecs;
		for (auto& light : lightInfors)
		{

			lightsRecs.emplace_back(light.rec());
		}
                //cvex::showRectangles("light", src, src, lightsRecs, cvex::MAGENTA, 1, _roi.tl());//显示灯条检测结果

#endif // DEBUG_DETECTION

		if (lightInfors.empty())
		{
			//std::cout << "there is no light" << endl;
			return 0;
		}
	}
	/*
		装甲板匹配
	*/
	{
		sort(lightInfors.begin(), lightInfors.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2)
		{
			return ld1.center.x < ld2.center.x;
		});
		vector<int> minRightIndices(lightInfors.size(), -1);
		for (size_t i = 0; i < lightInfors.size()-1; i++)
		{
			for (size_t j = i + 1; (j < lightInfors.size()); j++)
			{
				const LightDescriptor& leftLight = lightInfors[i];
				const LightDescriptor& rightLight = lightInfors[j];//定义左右灯条
				float angleDiff_ = abs(leftLight.angle - rightLight.angle);//左右矩形角度差
				float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);//差比率

				if (angleDiff_ > _param.light_max_angle_diff_ || LenDiff_ratio > _param.light_max_height_diff_ratio_)
					continue;
				float dis = cvex::distance(leftLight.center, rightLight.center);//两个灯条中心的距离
				float meanLen = (leftLight.length + rightLight.length) / 2;//平均长度
				float yDiff = abs(leftLight.center.y - rightLight.center.y);//坐标y的差值
				float yDiff_ratio = yDiff / meanLen;//
				float xDiff = abs(leftLight.center.x - rightLight.center.x);//坐标x的差值
				float xDiff_ratio = xDiff / meanLen;
				float ratio = dis / meanLen;

				if (yDiff_ratio > _param.light_max_y_diff_ratio_ ||
					xDiff_ratio < _param.light_min_x_diff_ratio_ || 
					ratio > _param.armor_max_aspect_ratio_ || ((yDiff - xDiff) / xDiff > 0.1)||
					ratio < _param.armor_min_aspect_ratio_)
				{
					continue;
				}
				/*
					用均值和方差去除中间太亮的图片（例如窗外的灯光等）
				*/
				Point p1, p2;
				int x1 = leftLight.center.x - leftLight.width/2+_roi.tl().x;
				int y1= leftLight.center.y - leftLight.length / 2+_roi.tl().y ;
				p1 = Point(x1, y1);
				int x2 = rightLight.center.x + rightLight.width / 2+ _roi.tl().x;
				int y2 = rightLight.center.y + rightLight.length / 2+ _roi.tl().y;
				p2 = Point(x2, y2);
				bool a;
				Rect roi_rect = Rect(p1, p2);

				if (roi_rect.x < 0)
					roi_rect.x = 0;
				if (roi_rect.x + roi_rect.width > src.cols)
					roi_rect.width = src.cols - roi_rect.x;
				if(roi_rect.y<0)
					roi_rect.y = 0;
				if (roi_rect.y + roi_rect.height > src.rows)
					roi_rect.height = src.rows - roi_rect.y;
				if (roi_rect.height < 0 || roi_rect.width < 0)
					a = false;
				else
					a = true;
				if (a)
				{
					Mat roi, mean, stdDev;
					roi = src(roi_rect).clone();
					double avg, stddev;
					meanStdDev(roi, mean, stdDev);
					avg = mean.ptr<double>(0)[0];
					stddev = stdDev.ptr<double>(0)[0];
					if (avg > 66.66)
						continue;
				}

				int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR;//判断装甲板在视野范围内是否较大

				float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio - ratio, float(0)) : max(_param.armor_small_armor_ratio - ratio, float(0));
				float yOff = yDiff / meanLen;
				float rotationScore = -(ratiOff * ratiOff + yOff * yOff);//计算装甲板大小分数
				cv::cvtColor(frame, img_gray,CV_BGR2GRAY);
				ArmorDescriptor armor(leftLight, rightLight, armorType, img_gray, rotationScore, _param);
				_armors.emplace_back(armor);//_armors为识别出的装甲板
				break;
			}
		}
#ifdef  DEBUG_DETECTION
		vector<vector<Point>> armorVertexs;
		for (const auto & armor : _armors)
		{
			vector<Point> intVertex;
			for (const auto& point : armor.vertex)
			{
				intVertex.emplace_back(Point(point.x, point.y));
			}
			armorVertexs.emplace_back(intVertex);
		}
                //cvex::showContours("dector", src, src, armorVertexs, cvex::RED, 1, _roi.tl());//显示装甲板检测结果
#endif //  DEBUG_DETECTION

		if (_armors.empty())
		{
			//cout << "there are no armor" << endl;
			return 1;
		}
	}
	/*
		获取装甲板图片，在ubuntu上可保存图片
	*/
#ifdef GET_ARMOR_PICTURE
	_allCnt++;
	for (const auto & armor : _armors)
	{
		Mat regulatedFrontImg = armor.frontImg;
		//regulatedFrontImg = regulatedFrontImg(Rect(21, 0, 50, 50));
		/*imwrite("/home/nvidia/Documents/ArmorTrainingSample/" + to_string(_allCnt) + "_" + to_string(i) + ".bmp", regulatedFrontImg);
		i++;*/
		char str[100];
		sprintf_s(str, "H:\\RM\\vision program\\装甲板检测_20200202\\Project1\\Project1\\image\\armor_%d.jpg", _allCnt);
		cv::imwrite(str, regulatedFrontImg);
	}
#endif // GET_ARMOR_PIC
	/*
		选择打击装甲板
	*/

	/*
	清除假装甲板
	*/
	
	_armors.erase(remove_if(_armors.begin(), _armors.end(), [](ArmorDescriptor& i)
	{
		return !(i.isArmorPattern());
	}), _armors.end());

	if (_armors.empty())
	{
		_targetArmor.clear();
		return 2;
	}

	for (auto & armor : _armors)
	{
		armor.finalScore = armor.sizeScore + armor.distScore + armor.rotaScore;
	}

	std::sort(_armors.begin(), _armors.end(), [](const ArmorDescriptor & a, const ArmorDescriptor & b)
	{
		return a.finalScore > b.finalScore;
	});

	_targetArmor = _armors[0];
        _targetArmor.armor_center.x = (_targetArmor.vertex[0].x + _targetArmor.vertex[1].x + _targetArmor.vertex[2].x + _targetArmor.vertex[3].x)*0.25f + _roi.tl().x;
        _targetArmor.armor_center.y = (_targetArmor.vertex[0].y + _targetArmor.vertex[1].y + _targetArmor.vertex[2].y + _targetArmor.vertex[3].y)*0.25f + _roi.tl().y;
        //std::cout<< _targetArmor.armor_center.x<<endl;

#if PREDICT
        _targetArmor.armor_center_predict=predictor.predict(_targetArmor.armor_center.x,_targetArmor.armor_center.y,0);
        //std::cout<< _targetArmor.armor_center_predict.x<<endl;
#endif
	//return 3;
        _trackCnt++;
#ifdef  DEBUG_DETECTION
	vector<Point> intVertex;
	for (const auto& point : _targetArmor.vertex)
	{
		Point fuckPoint = point;
		intVertex.emplace_back(fuckPoint);
	}
	//cout << "x=" << _targetArmor.armor_center.x <<","<<"y="<< _targetArmor.armor_center.y<<endl;
	
    //circle(src, _targetArmor.armor_center, 1, cvex::RED, -1);//标记出击打中心点

    //circle(src, _targetArmor.armor_center_predict, 1, cvex::YELLOW, -1);//标记出卡尔曼滤波后的点
	
        cvex::showContour("target", src, src, intVertex, cvex::GREEN, -1, _roi.tl());//显示目标打击装甲板
	

#endif

	return 3;

}

int getArmorType()
{
	return _targetArmor.type;
}

std::vector<cv::Point2f> getarmorvertex()//获取装甲板四个角点的坐标，带入PNP计算
{
	vector<cv::Point2f> realVertex;
	for (int i = 0; i < 4; i++)
	{
                realVertex.emplace_back(Point2f(_targetArmor.vertex[i].x+_roi.tl().x,_targetArmor.vertex[i].y+_roi.tl().y));
	}
	return realVertex;
}

void init()
{

    //klamanfilter_init();
	angleParam.readFile(9);
	_solverPtr.init(angleParam);
        //status = DETECTING_STATE;
        if(!Serial.Open("/dev/ttyUSB0",B115200,0,true))
            {
		printf("open serial port error\n");
		//InitUartAccomplish=true;
	    }
	else
        {
                printf("open serial port success\n");
		InitUartAccomplish=true;
        }
}
/*
	图像读取线程
*/
void image_produce()
{
    string path = "/home/jiing/zhuangjiaban/1.MOV";

    VideoCapture cap(path);
	
	while (cap.read(srcImg))
	{
		//cap >> srcImg;
		
		if (srcImg.empty())
			continue;
		
		while (prdIdx - csmIdx >= BUFFER_SIZE);
		imagedata[prdIdx % BUFFER_SIZE].img = srcImg;
		imagedata[prdIdx % BUFFER_SIZE].frame++;
		++prdIdx;
		//imshow("yuantu", imagedata[prdIdx % BUFFER_SIZE].img);
		//waitKey(1);

	}
}
/*
//鼠标动态调参
*/
void mouse_debug()
{
	std::this_thread::sleep_for(std::chrono::milliseconds(3));
	while (!stop)
	{
		while (prdIdx - csmIdx == 0);
		double time0 = static_cast<double>(getTickCount());
		imagedata[csmIdx % BUFFER_SIZE].img.copyTo(src);
		++csmIdx;
		if (src.empty())
			continue;
		cv::resize(src, src, cv::Size(640, 480));
		cv::split(src, bgr_channel);
		if (_param.enemy_color == BLUE)
		{
			cv::subtract(bgr_channel[0], bgr_channel[1], color_light);
			cv::cvtColor(src, img_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
			namedWindow("binary_light", WINDOW_AUTOSIZE);
			namedWindow("binary_color", WINDOW_AUTOSIZE);
                        namedWindow("yuchuli", WINDOW_AUTOSIZE);
                        createTrackbar("brightness_threshold:", "yuchuli", &brightness_threshold, 255, trackbar, 0);
                        createTrackbar("color_diff:", "yuchuli", &color_diff, 255, trackbar, 0);
			trackbar(0, 0);
			/*cv::threshold(img_gray, binary_brightness_img, _param.brightness_threshold_blue, 255, CV_THRESH_BINARY);
			imshow("binary_light", binary_brightness_img);
			cv::threshold(color_light, binary_color_img, _param.blue_color_diff, 255, CV_THRESH_BINARY);
			cv::dilate(binary_color_img, binary_color_img, element);
			imshow("binary_color", binary_color_img);
			binbrightimg = binary_color_img & binary_brightness_img;
			cv::dilate(binbrightimg, binbrightimg, element);*/
			//imshow("yuchuli", binbrightimg);
		}

		else if (_param.enemy_color == RED)
		{
			cv::subtract(bgr_channel[0], bgr_channel[1], color_light);
			cv::cvtColor(src, img_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
			namedWindow("binary_light", WINDOW_AUTOSIZE);
			namedWindow("binary_color", WINDOW_AUTOSIZE);
			createTrackbar("brightness_threshold:", "binary_light", &brightness_threshold, 255, trackbar, 0);
			createTrackbar("color_diff:", "binary_color", &color_diff, 255, trackbar, 0);
			trackbar(0, 0);
			/*cv::threshold(img_gray, binary_brightness_img, _param.brightness_threshold_red, 255, CV_THRESH_BINARY);
			imshow("binary_light", binary_brightness_img);
			cv::threshold(color_light, binary_color_img, _param.red_color_diff, 255, CV_THRESH_BINARY);
			cv::dilate(binary_color_img, binary_color_img, element);
			imshow("binary_color", binary_color_img);
			binbrightimg = binary_color_img & binary_brightness_img;
			cv::dilate(binbrightimg, binbrightimg, element);*/
			//imshow("yuchuli", binbrightimg);
		}
	}
}
/*
	图像处理进程
*/
void image_process()
{
	float a = 0;
	float time1 = 0;
	std::this_thread::sleep_for(std::chrono::milliseconds(3));
	Mat roiImg;
	//_roi = Rect(Point(0, 0), src.size());
       // while (!InitUartAccomplish);//等待串口初始化
	while (!stop)
	{
		while (prdIdx - csmIdx == 0);
		double time0 = static_cast<double>(getTickCount());
		imagedata[csmIdx % BUFFER_SIZE].img.copyTo(src);
		++csmIdx;
		if (src.empty())
			continue;
		cv::resize(src, src, cv::Size(640, 480));
		Rect imgBound = Rect(cv::Point(0, 0), src.size());
		//_roi = Rect(Point(0, 0), _srcImg.size());

		if (armor_detect_flag == 3)
		{
			cv::Rect bRect = boundingRect(_targetArmor.vertex) + _roi.tl();
			bRect = cvex::scaleRect(bRect, Vec2f(3, 2));
			_roi = bRect & imgBound;
			roiImg = src(_roi).clone();
		}
		else
		{
			_roi = imgBound;
			roiImg = src.clone();
			_trackCnt = 0;
		}
		armor_detect_flag = detect(roiImg);
		//rectangle(src, _roi, cvex::YELLOW);
		//imshow("roi", src);

                //armorVetex = getarmorvertex();//获得装甲板四个角点的坐标，用于PNP解算
                int armortype = getArmorType();

                _solverPtr.setTarget_point(_targetArmor.armor_center, armortype);//单点解算，只计算角度
                //_solverPtr.setTarget_pnp(armorVetex, armortype);//PNP解算偏转角及距离
		int angleflag= _solverPtr.solve();
		if (angleflag != AngleSolver::ANGLE_ERROR)
		{
			_solverPtr.compensateGravity();
			targetAngle = _solverPtr.getAngle();
			distance_armor = _solverPtr.getDistance();
			//cout << "5"<< endl;
                        cout << "pitch=" << targetAngle[1] << ","<< "yaw=" << targetAngle[0]<<","<<"distance="<< distance_armor <<endl;
                        while(prdIdx_uart-csmIdx_uart);
                        uartdata[prdIdx_uart % BUFFER_SIZE_UART].yaw=targetAngle[0];
                        uartdata[prdIdx_uart % BUFFER_SIZE_UART].pitch=targetAngle[1];
                        uartdata[prdIdx_uart % BUFFER_SIZE_UART].distance=distance_armor;
                        ++prdIdx_uart;

                }
                cv::waitKey(1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		time0 = time0 * 1000;//计算处理一帧图像所用的时间。
		time1 += time0;//time1/a为平均处理一帧图像的时间
		a++;
		float avarage_time = time1 / a;
                //std::cout << avarage_time << "ms" << endl;

	}
}

/*
 * uart send data thread
*/
void send_data()
{
    float send_yaw;
    float send_pitch;
    float send_distance;
    while(1)
    {
        while (prdIdx_uart - csmIdx_uart == 0);
        send_yaw=uartdata[csmIdx_uart%BUFFER_SIZE_UART].yaw;
        send_pitch=uartdata[csmIdx_uart%BUFFER_SIZE_UART].pitch;
        send_distance=uartdata[csmIdx_uart%BUFFER_SIZE_UART].distance;
        ++csmIdx_uart;

        /*if(Serial.send(send_yaw,send_pitch,send_distance))
            Serial.restart();*/
        //Serial.send(send_yaw,send_pitch,send_distance);
		Serial.send_ttltest(send_yaw);


    }

}


int main()
{
	
	init();

        std::thread produce(image_produce);

#if MOUSE_DEBUG

        std::thread debug(mouse_debug);

#else

        std::thread process(image_process);
        std::thread send(send_data);


#endif
        produce.join();

#if MOUSE_DEBUG
        debug.join();
#else
        process.join();
        send.join();
#endif



	return 0;

}

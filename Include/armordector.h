#pragma once

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

int BLUE = 0;
int  GREEN = 1;
int RED = 2;
int UNKNOWN_ARMOR = 0;
int SMALL_ARMOR = 1;
int BIG_ARMOR = 2;


Ptr<ml::SVM> svm = ml::StatModel::load<ml::SVM>("svm2.xml");



/*
        定义装甲板参数
*/

struct ArmorParam
{
        //Pre-treatment
        int brightness_threshold_blue;//灰度图二值化阈值
        int brightness_threshold_red;
        int color_threshold;
        float light_color_detect_extend_ratio;
        float red_color_diff;//通道之差二值化阈值
        float blue_color_diff;

        //Filter lights
        float light_min_area;
        float light_max_area;
        float light_max_angle;
        float light_min_size;
        float light_contour_min_solidity;
        float light_max_ratio;
        float light_min_ratio;

        //Filter pairs
        float light_max_angle_diff_;
        float light_max_height_diff_ratio_; // hdiff / max(r.length, l.length)
        float light_max_y_diff_ratio_;  // ydiff / max(r.length, l.length)
        float light_min_x_diff_ratio_;
        float light_max_x_diff_ratio_;

        //Filter armor
        float armor_big_armor_ratio;
        float armor_small_armor_ratio;
        float armor_min_aspect_ratio_;
        float armor_max_aspect_ratio_;

        //other params
        float sight_offset_normalized_base;
        float area_normalized_base;
        int enemy_color;
        int max_track_num = 3000;

        /*
        *	@Brief: 为各项参数赋默认值
        */
        ArmorParam()
        {
                //pre-treatment
                brightness_threshold_blue = 60;
                brightness_threshold_red = 60;
                color_threshold = 40;
                light_color_detect_extend_ratio = 1.1;
                red_color_diff = 40;
                blue_color_diff = 45;

                // Filter lights
                light_min_area = 8;
                light_max_area = 100;
                light_max_angle = 45.0;
                light_min_size = 5.0;
                light_contour_min_solidity = 0.3;
                light_min_ratio = 0.1;
                light_max_ratio = 0.75;

                // Filter pairs
                light_max_angle_diff_ = 5; //20
                light_max_height_diff_ratio_ = 0.3; //0.5
                light_max_y_diff_ratio_ = 2.0; //100
                light_min_x_diff_ratio_ = 0.5; //100
                light_max_x_diff_ratio_=2;

                // Filter armor
                armor_big_armor_ratio = 3.0;
                armor_small_armor_ratio = 2;
                //armor_max_height_ = 100.0;
                //armor_max_angle_ = 30.0;
                armor_min_aspect_ratio_ = 1.0;
                armor_max_aspect_ratio_ =5;

                //other params
                sight_offset_normalized_base = 200;
                area_normalized_base = 1000;
                enemy_color = BLUE;
        }
};

/*
        定义灯条具有的属性
*/
class LightDescriptor
{
public:
        LightDescriptor() {};
        LightDescriptor(const cv::RotatedRect& light)
        {
                width = light.size.width;
                length = light.size.height;
                center = light.center;
                angle = light.angle;
                area = light.size.area();
        }
        const LightDescriptor& operator =(const LightDescriptor& ld)
        {
                this->width = ld.width;
                this->length = ld.length;
                this->center = ld.center;
                this->angle = ld.angle;
                this->area = ld.area;
                return *this;
        }

        /*
        *	@Brief: return the light as a cv::RotatedRect object
        */
        cv::RotatedRect rec() const
        {
                return cv::RotatedRect(center, cv::Size2f(width, length), angle);
        }

public:
        float width;
        float length;
        cv::Point2f center;
        float angle;
        float area;
};


/*
        定义装甲板类及具有的属性
*/
class ArmorDescriptor
{
public:
        /*
        *	@Brief: Initialize with all 0
        */
        ArmorDescriptor()
        {
                rotaScore = 0;
                sizeScore = 0;
                vertex.resize(4);
                for (int i = 0; i < 4; i++)
                {
                        vertex[i] = cv::Point2f(0, 0);
                }
                type = UNKNOWN_ARMOR;
        }

        /*
        *	@Brief: calculate the rest of information(except for match&final score)of ArmroDescriptor based on:
                        l&r light, part of members in ArmorDetector, and the armortype(for the sake of saving time)
        *	@Calls: ArmorDescriptor::getFrontImg()
        */
        ArmorDescriptor(const LightDescriptor& lLight, const LightDescriptor& rLight, const int armorType, const cv::Mat& grayImg, const float rotationScore, ArmorParam param)
        {
                lightPairs[0] = lLight.rec();
                lightPairs[1] = rLight.rec();

                cv::Size exLSize(int(lightPairs[0].size.width), int(lightPairs[0].size.height * 2));
                cv::Size exRSize(int(lightPairs[1].size.width), int(lightPairs[1].size.height * 2));
                cv::RotatedRect exLLight(lightPairs[0].center, exLSize, lightPairs[0].angle);
                cv::RotatedRect exRLight(lightPairs[1].center, exRSize, lightPairs[1].angle);

                cv::Point2f pts_l[4];
                exLLight.points(pts_l);
                //cv::Point2f upper_l = pts_l[2];
                //cv::Point2f lower_l = pts_l[3];
                cv::Point2f upper_l = pts_l[1];
                cv::Point2f lower_l = pts_l[0];

                cv::Point2f pts_r[4];
                exRLight.points(pts_r);
                //cv::Point2f upper_r = pts_r[1];
                //cv::Point2f lower_r = pts_r[0];
                cv::Point2f upper_r = pts_r[2];
                cv::Point2f lower_r = pts_r[3];

                vertex.resize(4);
                vertex[0] = upper_l;
                vertex[1] = upper_r;
                vertex[2] = lower_r;
                vertex[3] = lower_l;//将图像中绿色矩形的四个顶点坐标赋值给vertex

                //set armor type
                type = armorType;

                //get front view
                getFrontImg(grayImg);
                rotaScore = rotationScore;

                // calculate the size score
                float normalized_area = contourArea(vertex) / 1000;//_param.area_normalized_base
                sizeScore = exp(normalized_area);

                // calculate the distance score
                Point2f srcImgCenter(grayImg.cols / 2, grayImg.rows / 2);
                float sightOffset = cvex::distance(srcImgCenter, cvex::crossPointOf(array<Point2f, 2>{vertex[0], vertex[2]}, array<Point2f, 2>{vertex[1], vertex[3]}));
                distScore = exp(-sightOffset / 200);//_param.sight_offset_normalized_base





        }

        /*
        *	@Brief: empty the object
        *	@Called :ArmorDetection._targetArmor
        */
        void clear()
        {
                rotaScore = 0;
                sizeScore = 0;
                distScore = 0;
                finalScore = 0;
                for (int i = 0; i < 4; i++)
                {
                        vertex[i] = cv::Point2f(0, 0);
                }
                type = UNKNOWN_ARMOR;
        }

        /*
        *	@Brief: get the front img(prespective transformation) of armor
        *	@Inputs: grayImg of roi
        *	@Outputs: store the front img to ArmorDescriptor's public
        */
        void getFrontImg(const cv::Mat& grayImg)
        {
                using cvex::distance;
                const Point2f&
                        tl = vertex[0],
                        tr = vertex[1],
                        br = vertex[2],
                        bl = vertex[3];

                int width, height;
                if (type == BIG_ARMOR)
                {
                        //width = 92;
                        width = 50;
                        height = 50;
                }
                else
                {
                        width = 50;
                        height = 50;
                }

                Point2f src[4]{ Vec2f(tl), Vec2f(tr), Vec2f(br), Vec2f(bl) };
                Point2f dst[4]{ Point2f(0.0, 0.0), Point2f(width, 0.0), Point2f(width, height), Point2f(0.0, height) };
                const Mat perspMat = getPerspectiveTransform(src, dst);
                cv::warpPerspective(grayImg, frontImg, perspMat, Size(width, height));
        }

        /*
        *
                利用SVM判断是否为装甲板
        */
        bool isArmorPattern() const
        {
                //    // cut the central part of the armor
                Mat regulatedImg;
                if (type == BIG_ARMOR)
                {
                        //regulatedImg = frontImg(Rect(21, 0, 50, 50));
                        regulatedImg = frontImg;
                }
                else
                {
                        regulatedImg = frontImg;
                }

                //resize(regulatedImg, regulatedImg, Size(regulatedImg.size().width / 2, regulatedImg.size().height / 2));
                // copy the data to make the matrix continuous
                Mat temp;
                regulatedImg.copyTo(temp);
                Mat data = temp.reshape(1, 1);
                data.convertTo(data, CV_32FC1);
                if (svm.empty())
                {
                        std::cout << "The model load failed" << endl;
                }

                int result = (int)svm->predict(data);
                if (result == 1)
                        return true;
                else
                        return false;

        }

public:
        std::array<cv::RotatedRect, 2> lightPairs; //0 left, 1 right
        float sizeScore;		//S1 = e^(size)
        float distScore;		//S2 = e^(-offset)
        float rotaScore;		//S3 = -(ratio^2 + yDiff^2)
        float finalScore;
        Point2f armor_center;//变换ROI后在图像中显示装甲板中心点
        Point2f armor_center_real;//实际中心点，要考虑ROI变化
        Point2f armor_center_predict;//预测中心点
        std::vector<cv::Point2f> vertex; //four vertex of armor area, lihgt bar area exclued!!
        cv::Mat frontImg; //front img after prespective transformation from vertex

        int type;
};



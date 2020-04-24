#pragma once

#include"/home/jiing/zhuangjiaban/include/kalmanfilter.h"
#include<iostream>
#include<math.h>
#include<opencv2/opencv.hpp>

#include<thread>
#include<atomic>

using namespace std;
using namespace cv;

class Predictor{
public:
    Predictor()
    {
        int stateSize=4;
        int measureSize=2;
        Eigen::MatrixXd A(stateSize, stateSize);
        A << 1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1 ;

        Eigen::MatrixXd H(measureSize, stateSize);
        H << 1, 0, 0, 0,
                0, 1, 0, 0;

        Eigen::MatrixXd P(stateSize, stateSize);
        P << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;

        Eigen::MatrixXd Q(stateSize, stateSize);
        Q << 5, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 5, 0,
                0, 0, 0, 1;

        Eigen::MatrixXd R(measureSize, measureSize);
        R << 150, 0,
                0, 150;


        KF.init(stateSize, measureSize, A, P, R, Q, H);

        x.resize(stateSize);
        x << 0, 0, 0, 0;
    }
    ~Predictor()
    {

    }
public:
    cv::Point2f predict(float centerx,float centery,float time)
    {
        cv::Point2f predictPoint = kalmanPredict(centerx, centery, time);
        return predictPoint;
    }
private:
    cv::Point2f kalmanPredict(float centerx,float centery,float time)
    {
        /*if (abs(x(0) - centerx) > 200) {
                x(0) = (double)centerx;
                x(2) = 0;
        }
        if (abs(x(2) - centery) > 200) {
                x(1) = (double)centery;
                x(3) = 0;
        }*/
        Eigen::VectorXd z(2);
        Eigen::VectorXd output;

        z<<(float)centerx,(float) centery;
        KF.predict(x);
        KF.update(x,z);

        float predictx=x(0) + (time + 5) * x(2);
        float predicty=x(1) + (time + 5) * x(3);
        /*float predictx=x(0);
        float predicty=x(1);*/



        return cv::Point2f(predictx,predicty);
    }
private:
    EigenKalman::KalmanFilter KF;
    Eigen::VectorXd x;

};

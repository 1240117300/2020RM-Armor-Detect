#pragma once

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


/*
        定义角度解算参数结构体
*/
struct AngleSolverParam
{

        cv::Mat CAM_MATRIX ;		//Camera Matrix
        cv::Mat DISTORTION_COEFF;	//Distortion matrix
        //the unit of vector is mm

        double Y_DISTANCE_BETWEEN_GUN_AND_CAM = 0;//If gun is under the cam, this variable is positive.

        /*
        * @brief set the params of camera
        * @param Input int id is the id of camera.you should write a xml including your camera's params.
        */
        void readFile(const int id)
        {
                cv::FileStorage fsread("angle_solver_params.xml", cv::FileStorage::READ);
                if (!fsread.isOpened())
                {
                        std::cerr << "failed to open xml" << std::endl;
                        return;
                }
                fsread["Y_DISTANCE_BETWEEN_GUN_AND_CAM"] >> Y_DISTANCE_BETWEEN_GUN_AND_CAM;

#ifdef DEBUG
                std::cout << Y_DISTANCE_BETWEEN_GUN_AND_CAM << std::endl;
#endif // DEBUG

                switch (id)
                {
                case 0:
                {
                        fsread["CAMERA_MARTRIX_0"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_0"] >> DISTORTION_COEFF;
                        return;
                }
                case 1:
                {
                        fsread["CAMERA_MARTRIX_1"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_1"] >> DISTORTION_COEFF;
                        return;
                }
                case 2:
                {
                        fsread["CAMERA_MARTRIX_2"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_2"] >> DISTORTION_COEFF;
                        return;
                }
                case 3:
                {
                        fsread["CAMERA_MARTRIX_3"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_3"] >> DISTORTION_COEFF;
                        return;
                }
                case 4:
                {
                        fsread["CAMERA_MARTRIX_4"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_4"] >> DISTORTION_COEFF;
                        return;
                }
                case 5:
                {
                        fsread["CAMERA_MARTRIX_OFFICIAL"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_OFFICIAL"] >> DISTORTION_COEFF;
                        return;
                }
                case 6:
                {
                        fsread["CAMERA_MARTRIX_6"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_6"] >> DISTORTION_COEFF;
                        return;
                }
                case 7:
                {
                        fsread["CAMERA_MARTRIX_7"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_7"] >> DISTORTION_COEFF;
                        return;
                }
                case 8:
                {
                        fsread["CAMERA_MARTRIX_8"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_8"] >> DISTORTION_COEFF;
                        return;
                }
                case 9:
                {
                        fsread["CAMERA_MARTRIX_9"] >> CAM_MATRIX;
                        fsread["DISTORTION_COEFF_9"] >> DISTORTION_COEFF;
                        return;
                }
                default:
                        std::cout << "wrong cam number given." << std::endl;
                        return;
                }
        }


        vector<cv::Point3f> POINT_3D_OF_ARMOR_BIG = std::vector<cv::Point3f>
        {

                        cv::Point3f(-105, -30, 0),	//tl
                        cv::Point3f(105, -30, 0),	//tr
                        cv::Point3f(105, 30, 0),	//br
                        cv::Point3f(-105, 30, 0)	//bl
        };
        vector<cv::Point3f> POINT_3D_OF_RUNE = std::vector<cv::Point3f>
        {
                cv::Point3f(-370, -220, 0),
                cv::Point3f(0, -220, 0),
                cv::Point3f(370, -220, 0),
                cv::Point3f(-370, 0, 0),
                cv::Point3f(0, 0, 0),
                cv::Point3f(370, 0, 0),
                cv::Point3f(-370, 220, 0),
                cv::Point3f(0, 220, 0),
                cv::Point3f(370, 220, 0)
        };

        vector<cv::Point3f> POINT_3D_OF_ARMOR_SMALL = std::vector<cv::Point3f>
        {
                cv::Point3f(-65, -35, 0),	//tl
                cv::Point3f(65, -35, 0),	//tr
                cv::Point3f(65, 35, 0),		//br
                cv::Point3f(-65, 35, 0)		//bl
        };
};
/**
*	solve by PNP, that is, using four points to detect the angle and distance.
*	It's not very useful if the armor is far.If it' far try solve by one point
*/

class AngleSolver
{
private:
        AngleSolverParam _params;
        cv::Mat _rVec = cv::Mat::zeros(3, 1, CV_64FC1);//init rvec
        cv::Mat _tVec = cv::Mat::zeros(3, 1, CV_64FC1);//init tvec
        std::vector<cv::Point2f> point_2d_of_armor;
        std::vector<cv::Point2f> point_2d_of_rune;
        int angle_solver_algorithm = 0;//if 1 ,using PNP solution, if 0 using OnePoint solution.
        cv::Point2f centerPoint;
        std::vector<cv::Point2f> target_nothing;
        double _xErr, _yErr, _euclideanDistance;

        int enemy_type = 1;//1 is the big armor,0 is the small armor
        double _bullet_speed = 22000;

        cv::Mat _cam_instant_matrix;// a copy of camera instant matrix

public:
        AngleSolver()
        {
                
        }

        AngleSolver(const AngleSolverParam& AngleSolverParam)
        {
                _params = AngleSolverParam;
                _cam_instant_matrix = _params.CAM_MATRIX.clone();
                
        }

        /*
          Initialize with parameters
         */
        void init(const AngleSolverParam& AngleSolverParam)
        {
                _params = AngleSolverParam;
                _cam_instant_matrix = _params.CAM_MATRIX.clone();
        }

        enum AngleFlag
        {
                ANGLE_ERROR = 0,                //an error appeared in angle solution
                ONLY_ANGLES = 1,		//only angles is avilable
                TOO_FAR = 2,			//the distance is too far, only angles is avilable
                ANGLES_AND_DISTANCE = 3		//distance and angles are all avilable and correct
        };


        /*
        * @brief set the 2D center point or corners of armor, or the center of buff as target
        * @param Input armorPoints/centerPoint
        */
        void setTarget_pnp(const std::vector<cv::Point2f> objectPoints, int objectType)  //set corner points for PNP
        {
                if (objectType == 1 || objectType == 2)
                {
                        if (angle_solver_algorithm == 0 || angle_solver_algorithm == 2)
                        {
                                angle_solver_algorithm = 1;
                                std::cout << "algorithm is reset to PNP Solution" << endl;
                        }
                        point_2d_of_armor = objectPoints;
                        if (objectType == 1)
                                enemy_type = 0;
                        else
                                enemy_type = 1;
                        return;
                }
        }
        void setTarget_point(const cv::Point2f Center_of_armor, int objectPoint)//set center points
        {
                if (angle_solver_algorithm == 1 || angle_solver_algorithm == 2)
                {
                        angle_solver_algorithm = 0;
                        std::cout << "algorithm is reset to One Point Solution" << endl;
                }
                centerPoint = Center_of_armor;
            
        }



        /*
        * @brief slove the angle by selected algorithm
        */
        AngleFlag solve()
        {
                if (angle_solver_algorithm == 1)
                {
                        if (enemy_type == 1)
                        {
                                solvePnP(_params.POINT_3D_OF_ARMOR_BIG, point_2d_of_armor, _cam_instant_matrix, _params.DISTORTION_COEFF, _rVec, _tVec, false, SOLVEPNP_ITERATIVE);//opencv3
                               
                        }
                        if (enemy_type == 0)
                        {
                                solvePnP(_params.POINT_3D_OF_ARMOR_SMALL, point_2d_of_armor, _cam_instant_matrix, _params.DISTORTION_COEFF, _rVec, _tVec, false, SOLVEPNP_ITERATIVE);//opencv3
                                
                        }

                        _tVec.at<double>(1, 0) -= _params.Y_DISTANCE_BETWEEN_GUN_AND_CAM;
                        _xErr = atan(_tVec.at<double>(0, 0) / _tVec.at<double>(2, 0)) / 2 / CV_PI * 360;
                        _yErr = atan(_tVec.at<double>(1, 0) / _tVec.at<double>(2, 0)) / 2 / CV_PI * 360;
                        _euclideanDistance = sqrt(_tVec.at<double>(0, 0)*_tVec.at<double>(0, 0) + _tVec.at<double>(1, 0)*_tVec.at<double>(1, 0) + _tVec.at<double>(2, 0)* _tVec.at<double>(2, 0));
                        /*if (_euclideanDistance >= 8500)
                        {
                                return TOO_FAR;
                        }*/
                        return ANGLES_AND_DISTANCE;
                }
                if (angle_solver_algorithm == 0)
                {
                        double fx = _cam_instant_matrix.at<double>(0, 0);
                        double fy = _cam_instant_matrix.at<double>(1, 1);
                        double cx = _cam_instant_matrix.at<double>(0, 2);
                        double cy = _cam_instant_matrix.at<double>(1, 2);
                        double k1 = _params.DISTORTION_COEFF.at<double>(0, 0);
                        double k2 = _params.DISTORTION_COEFF.at<double>(1, 0);
                        double p1 = _params.DISTORTION_COEFF.at<double>(2, 0);
                        double p2 = _params.DISTORTION_COEFF.at<double>(3, 0);
                        Point2f pnt;
                        vector<cv::Point2f>in;
                        vector<cv::Point2f>out;
                        in.push_back(Point2f(centerPoint.x, centerPoint.y));
                        undistortPoints(in, out, _cam_instant_matrix, _params.DISTORTION_COEFF, noArray(), _cam_instant_matrix);
                        pnt = out.front();
                        double rxNew = (pnt.x - cx) / fx;
                        double ryNew = (pnt.y - cy) / fy;
                        double y_ture = ryNew - _params.Y_DISTANCE_BETWEEN_GUN_AND_CAM / 1000;
                        _xErr = atan(rxNew) / 2 / CV_PI * 360;
                        _yErr = atan(y_ture) / 2 / CV_PI * 360;


                        return ONLY_ANGLES;
                }
          
                return ANGLE_ERROR;
        }

        /*
        *      z: direction of the shooter
        *     /
        *  O /______x
        *    |
        *    |
        *    y
        */
        void compensateOffset()//偏移补偿
        {
                /* z of the camera COMS */
                const auto offset_z = 115.0;
                const auto& d = _euclideanDistance;
                const auto theta_y = _xErr / 180 * CV_PI;
                const auto theta_p = _yErr / 180 * CV_PI;
                const auto theta_y_prime = atan((d*sin(theta_y)) / (d*cos(theta_y) + offset_z));
                const auto theta_p_prime = atan((d*sin(theta_p)) / (d*cos(theta_p) + offset_z));
                const auto d_prime = sqrt(pow(offset_z + d * cos(theta_y), 2) + pow(d*sin(theta_y), 2));
                _xErr = theta_y_prime / CV_PI * 180;
                _yErr = theta_p_prime / CV_PI * 180;
                _euclideanDistance = d_prime;
        }
        void compensateGravity()//重力补偿
        {
                const auto& theta_p_prime = _yErr / 180 * CV_PI;
                const auto& d_prime = _euclideanDistance;
                const auto& v = _bullet_speed;
                const auto theta_p_prime2 = atan((sin(theta_p_prime) - 0.5*9.8*d_prime / pow(v, 2)) / cos(theta_p_prime));
                _yErr = theta_p_prime2 / CV_PI * 180;
        }

        /*
        * @brief get the angle solved
        */
        const cv::Vec2f getAngle()
        {
                return cv::Vec2f(_xErr, _yErr);
        }
        /*
        * @brief get the distance between the camera and the target
        */
        double getDistance()
        {
                return _euclideanDistance;
        }

};


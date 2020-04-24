#include <opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include<opencv2/ml/ml.hpp>
#include<stdio.h>
#include<io.h>

using namespace std;
using namespace cv;

	void getFiles(string path, vector<string>& files);
	void get_1(Mat& trainingImages, vector<int>& trainingLabels);
	void get_0(Mat& trainingImages, vector<int>& trainingLabels);

	int main()
	{
		//获取训练数据
		Mat classes;
		Mat trainingData;
		Mat trainingImages;
		vector<int> trainingLabels;
		get_1(trainingImages, trainingLabels);
		get_0(trainingImages, trainingLabels);
		Mat(trainingImages).copyTo(trainingData);
		trainingData.convertTo(trainingData, CV_32FC1);
		Mat(trainingLabels).copyTo(classes);
		//配置SVM训练器参数
		/*
			opencv3
		*/
		Ptr<ml::SVM> SVM_params = ml::SVM::create();
		SVM_params->setType(ml::SVM::C_SVC);
		SVM_params->setKernel(ml::SVM::LINEAR);  //核函数

		SVM_params->setDegree(0);
		SVM_params->setGamma(1);
		SVM_params->setCoef0(0);
		SVM_params->setC(1);
		SVM_params->setNu(0);
		SVM_params->setP(0);
		SVM_params->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));


		Ptr<ml::TrainData> tData = ml::TrainData::create(trainingData,ml::SampleTypes::ROW_SAMPLE, classes);
		//保存模型
		SVM_params->train(tData);
		SVM_params->save("svm.xml");

		/*
			opencv2
		*/
		/*SVM MY_SVM;
		cv::SVMParams params;
		params.svm_type = SVM::C_SVC;
		params.kernel_type = SVM::LINEAR;
		params.C = 0.01;
		params.p = 0.1;
		params.nu = 0.5;
		params.gamma = 0;
		params.degree = 3;
		params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
		MY_SVM.train(trainingData, classes, Mat(), Mat(), params);
		MY_SVM.save("svm_opencv2.xml");*/
		cout << "训练好了！！！" << endl;
		getchar();
		return 0;
	}
	void getFiles(string path, vector<string>& files)
	{
		intptr_t   hFile = 0;
		struct _finddata_t fileinfo;
		string p;
		int i = 30;
		if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		{

			do
			{
				if ((fileinfo.attrib &  _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
						getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
				else
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}

			} while (_findnext(hFile, &fileinfo) == 0);

			_findclose(hFile);
		}
	}
	void get_1(Mat& trainingImages, vector<int>& trainingLabels)
	{
			char * filePath = (char*)"H:\\RM\\装甲板图片\\8";
			vector<string> files;
			getFiles(filePath, files);
			int number = files.size();
			for (int i = 0; i < number; i++)
			{
				Mat SrcImage = imread(files[i].c_str());
				cvtColor(SrcImage, SrcImage, CV_BGR2GRAY);
				resize(SrcImage, SrcImage, cv::Size(50, 50));
				//imshow("main", SrcImage);
				SrcImage = SrcImage.reshape(1, 1);
				trainingImages.push_back(SrcImage);
				trainingLabels.push_back(1);
			}
		
	}
	void get_0(Mat& trainingImages, vector<int>& trainingLabels)
	{

		char * filePath = (char*)"H:\\RM\\装甲板图片\\7";
		vector<string> files;
		getFiles(filePath, files);
		int number = files.size();
		for (int i = 0; i < number; i++)
		{
			Mat SrcImage = imread(files[i].c_str());
			cvtColor(SrcImage, SrcImage, CV_BGR2GRAY);
			resize(SrcImage, SrcImage, cv::Size(50, 50));
			//imshow("main", SrcImage);
			SrcImage = SrcImage.reshape(1, 1);
			trainingImages.push_back(SrcImage);
			trainingLabels.push_back(0);
		}
		
	}




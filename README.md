# 2020RM-Armor-Detect
This project is the detection code of the sentinel armor of the Jianfeng team of Shenyang Jianzhu University

# Platform:
1.ubuntu16.04

2.opencv3.4.1

3.cmake

# How to run the code
1.Modify the part of CmakeLists.txt that contains the file path and change its path to the path of the file in your ubuntu system

2.Same as 1, Modify the path in the program

3.In terminal, using cmake .or cmake ..to create the makefile of the project

4.In terminal, using the make to compile the project and it will create a executable file

5.In termianl, using ./+the name of executable file to run the project as ./zhuangjiaban

# The Explanation of file
1. The Include directory,contains the Head files of the project.

2.The angle_solver_params.xml contains Contains camera calibration data

3. The svm2.xml is the model trained by SVM, which code is showed in SVM train code



#include <iostream>
#include "MTCNN.h"
#include "opencv2/opencv.hpp"
#include <time.h>


using namespace std;
using namespace cv;

//int main() {
//
//    vector<string> model_file = {
//            "../model/det1.prototxt",
//            "../model/det2.prototxt",
//            "../model/det3.prototxt"
////            "../model/det4.prototxt"
//    };
//
//    vector<string> trained_file = {
//            "../model/det1.caffemodel",
//            "../model/det2.caffemodel",
//            "../model/det3.caffemodel"
////            "../model/det4.caffemodel"
//    };
//
//    MTCNN mtcnn(model_file, trained_file);
//
//    vector<Rect> rectangles;
//    string img_path = "../result/trump.jpg";
//    Mat img = imread(img_path);
//
//    mtcnn.detection(img, rectangles);
//
//    std::cout << "Hello, World!" << std::endl;
//    return 0;
//}

int main(int   argc,   char*   argv[]) {

    string filename;
    if(argc!=2)
    {
        cout<<"usage: mtcnn_train direction (middle/left/right)"<<endl;
         return 0;
    }
    cout<<argv[1]<<endl;
    if(strcmp(argv[1],"left" ) == 0)
        filename="/home/zmz/MTCNN-master/data/left/";
    else if(strcmp(argv[1],"middle" ) == 0)
        filename="/home/zmz/MTCNN-master/data/middle/";
    else if(strcmp(argv[1],"right" ) == 0)
        filename="/home/zmz/MTCNN-master//data/right/";
    else
    {
        cout<<"error: invalid variable "<<endl;
         return 0;
    }
    //the vector used to input the address of the net model
    vector<string> model_file = {
            "../model/det1.prototxt",
            "../model/det2.prototxt",
            "../model/det3.prototxt"
//            "../model/det4.prototxt"
    };

    //the vector used to input the address of the net parameters
    vector<string> trained_file = {
            "../model/det1.caffemodel",
            "../model/det2.caffemodel",
            "../model/det3.caffemodel"
//            "../model/det4.caffemodel"
    };

    MTCNN mtcnn(model_file, trained_file);

    VideoCapture cap(1);
//    VideoCapture cap("../../SuicideSquad.mp4");

//    VideoWriter writer;
//    writer.open("../result/SuicideSquad.mp4",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);

    Mat img;
    int frame_count = 0;
    while(cap.read(img))
    {
        vector<Rect> rectangles;
        vector<float> confidences;
        std::vector<std::vector<cv::Point>> alignment;
        mtcnn.detection(img, rectangles, confidences, alignment);

        string time;
        for(int i = 0; i < rectangles.size(); i++)
        {
                if(rectangles[i].x<=0 ||rectangles[i].width<=0||rectangles[i].x+rectangles[i].width>img.cols
                        ||rectangles[i].y<=0 ||rectangles[i].height<=0||rectangles[i].y+rectangles[i].height>img.rows)
                    continue;

                time_t t = std::time(NULL);
                char tmp[32];
                strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S",localtime(&t));
                time=tmp;



                Mat image_cut = Mat(img, rectangles[i]);      //从img中按照rect进行切割，此时修改image_cut时image中对应部分也会修改，因此需要copy
                Mat image_copy = image_cut.clone();   //clone函数创建新的图片
                char str[20];
                sprintf(str,"%d",i);
                char name_count[20];
                sprintf(name_count,"%d",frame_count);
                imshow(str,image_copy);
                waitKey(1);

                imwrite( filename+time+"_"+name_count+"_"+str+".jpg", image_copy);   //保存mat格式的图片成jpg格式，或者png，bmp格式，文件大小依次增大
        }

        for(int i = 0; i < rectangles.size(); i++)
        {
            int green = confidences[i] * 255;
            int red = (1 - confidences[i]) * 255;



            rectangle(img, rectangles[i], cv::Scalar(0, green, red), 3);
            for(int j = 0; j < alignment[i].size(); j++)
            {
                cv::circle(img, alignment[i][j], 5, cv::Scalar(255, 255, 0), 3);
            }


        }

        frame_count++;
        cv::putText(img, std::to_string(frame_count), cvPoint(3, 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
//        writer.write(img);
        imshow("Live", img);
        char c;
        c=waitKey(1);
        if(c=='q')
            break;
    }
    cap.release();
    return 0;
}



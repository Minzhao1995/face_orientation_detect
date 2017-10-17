#include <iostream>
#include "MTCNN.h"
#include "vggface.h"
#include "opencv2/opencv.hpp"

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

int main() {
    string vggface_model_file   = "../model/VGG_FACE_deploy.prototxt";
    string vggface_trained_file =  "../model/vgg_face_iter_271.caffemodel";
    string vggface_label_file   =  "../model/label.txt";
    Classifier classifier(vggface_model_file, vggface_trained_file, vggface_label_file);

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

        for(int i = 0; i < rectangles.size(); i++)
        {
            if(rectangles[i].x<=0 ||rectangles[i].width<=0||rectangles[i].x+rectangles[i].width>img.cols
                    ||rectangles[i].y<=0 ||rectangles[i].height<=0||rectangles[i].y+rectangles[i].height>img.rows)
                continue;
            int green = confidences[i] * 255;
            int red = (1 - confidences[i]) * 255;
            rectangle(img, rectangles[i], cv::Scalar(0, green, red), 3);
            for(int j = 0; j < alignment[i].size(); j++)
            {
                cv::circle(img, alignment[i][j], 5, cv::Scalar(255, 255, 0), 3);
            }


           // cv::Mat img2 = cv::imread("/home/zmz/MTCNN-master/detection/1.jpg", -1);
            //CHECK(!img2.empty()) << "Unable to decode image " <<endl;

            cv::Mat face_img;
             face_img = Mat(img, rectangles[i]);
            cv::resize( face_img, face_img, cv::Size(224,224));

            std::vector<Prediction> predictions = classifier.Classify(face_img);
              Prediction p = predictions[0];
              std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                        << p.first << "\"" << std::endl;
              cv::putText(img,  p.first, cvPoint(60, 130),
                          cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0, 0, 255), 1, CV_AA);
        }

        frame_count++;
        cv::putText(img, std::to_string(frame_count), cvPoint(3, 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);

//        writer.write(img);
        imshow("Live", img);
        waitKey(1);
    }

    return 0;
}



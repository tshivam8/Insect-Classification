#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int useEqualize = 0;
int th1, blursSize;
const std::string winName = "Tuna fish";
cv::Mat src, dst;
cv::Mat brightness;

void onTunaFishTrackbar(int, void*)
{
    cv::Mat hist, histImg, tmp;
    brightness.copyTo(tmp);

    if (blursSize >= 3)
    {
        blursSize += (1 - blursSize % 2);
        cv::GaussianBlur(tmp, tmp, cv::Size(blursSize, blursSize), 0);
    }
    if (useEqualize)
        cv::equalizeHist(tmp, tmp);

    cv::imshow("Brightness Preprocess", tmp);

    // threshold to select dark tuna
    cv::threshold(tmp, tmp, th1, 255, cv::THRESH_BINARY_INV);
    cv::imshow(winName, tmp);

    // find external contours ignores holes in the fish
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // draw all contours and select the largest
    src.copyTo(dst);
    double maxDim = 0;
    int largest = -1;
    for (int i = 0; i < contours.size(); i++)
    {
        // draw all contours in red
        cv::drawContours(dst, contours, largest, cv::Scalar(0, 0, 255), 1);
        int dim = contours[i].size(); //area is more accurate but more expensive
        //double dim = contourArea(contours[i]);
        //double dim = cvRound(arcLength(contours[i], true));
        if (dim > maxDim)
        {
            maxDim = dim;
            largest = i;
        }
    }

    //The tuna as binary mask
    cv::Mat fishMask = cv::Mat::zeros(src.size(), CV_8UC1);
    //The tuna as contour
    vector<cv::Point> theFish;
    if (largest >= 0)
    {
        theFish = contours[largest];
        // draw selected contour in bold green
        cv::polylines(dst, theFish, true, cv::Scalar(0, 255, 0), 2);
        // draw the fish into its mask
        cv::drawContours(fishMask, contours, largest, 255, -1);
    }
    cv::imshow("Result Fish Mask", fishMask);
    cv::imshow("Result Contour", dst);
}

int main(int argc, char* argv[])
{
    src = cv::imread("1.jpg");
    if (src.empty())
    {
        cout << endl
            << "ERROR! Unable to read the image" << endl
            << "Press a key to terminate";
        cin.get();
        return 1;

    }

    imshow(winName, src);
    imshow("Src", src);

    cvtColor(src, dst, COLOR_BGR2HSV);
    vector<cv::Mat > hsv_planes;
    split(dst, hsv_planes);
    //hue = hsv_planes[0];
    //saturation = hsv_planes[1];
    brightness = hsv_planes[2];

    // default settings for params
    useEqualize = 1;
    blursSize = 21;
    th1 = 33.0 * 255 / 100; //tuna is dark than select dark zone below 33% of full range
    cv::createTrackbar("Equalize", winName, &useEqualize, 1, onTunaFishTrackbar, 0);
    cv::createTrackbar("Blur Sigma", winName, &blursSize, 100, onTunaFishTrackbar, 0);
    cv::createTrackbar("Threshold", winName, &th1, 255, onTunaFishTrackbar, 0);

    onTunaFishTrackbar(0, 0);

    cv::waitKey(0);
    return 0;

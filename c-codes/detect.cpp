
#include "livenessdetector.h"

#include <fstream>
#include <iostream>
#include <cv_common.h>
#include <cstdlib>
#include <protector.h>
#include <opencv2/opencv.hpp>
#include <cv_liveness_internal.h>
#include <cv_common_internal.h>
#include <unistd.h>
#include <android/log.h>
#include <cv_common.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace{

#define  LOGI(...) __android_log_print(ANDROID_LOG_INFO, "========= Info =========   ", __VA_ARGS__)

#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "========= Error =========   ", __VA_ARGS__)

#define  LOGD(...)  __android_log_print(ANDROID_LOG_INFO, "========= Debug =========   ", __VA_ARGS__)

#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN, "========= Warn =========   ", __VA_ARGS__)

//const char k_license_file[] = "SENSETIME_1F2E5FAA-DBF7-41FA-AC2A-D6D8A52F18CD.lic";
const char k_color_file[] = "/storage/emulated/0/Documents/c%d.jpg";
const char k_gray_file[] = "/storage/emulated/0/Documents/g%d.jpg";

int AddLicense(const char* licenseFile) {
	std::ifstream file(licenseFile);
	if(!file) {
        LOGE("read license file failed!");
		return -1;
	}

	std::string license_content((std::istreambuf_iterator<char>(file)),
								std::istreambuf_iterator<char>());

	int state = sdk_protector_has_license("SgEngineering_Liveness");
	 if (state != CV_OK) {
         LOGE("has license failed! %d, [%s]", state, licenseFile);
	 	state = sdk_protector_add_license("SgEngineering_Liveness", license_content.c_str(), NULL);
	 	if (state != CV_OK) {
	 		LOGE("add license failed! %d, [%s]", state, licenseFile);
            return -1;
	 	}
	 }
	 return 0;
}

}

LivenessDetector::LivenessDetector():
    m_detect_model(nullptr),m_detect_handle(nullptr),
    m_align_model(nullptr),m_track_handle(nullptr),
    m_selector_model(nullptr),m_selector_handle(nullptr),
    m_hackness_model(nullptr),m_hackness_handle(nullptr),
    m_best_selector_score(0.0),m_imageIndex(0)
{

}


LivenessDetector::~LivenessDetector()
{
    if(nullptr != m_hackness_handle){
        cv_liveness_antispoofing_general_destroy(m_hackness_handle);
    }

    if(nullptr != m_hackness_model){
        cv_common_unload_model(m_hackness_model);
    }

    if(nullptr != m_selector_handle){
        cv_liveness_verification_frame_selector_destroy(m_selector_handle);
    }

    if(nullptr != m_selector_model){
        cv_common_unload_model(m_selector_model);
    }

    if(nullptr != m_track_handle){
        cv_common_tracking_compact_destroy(m_track_handle);
    }

    if(nullptr != m_align_model){
        cv_common_unload_model(m_align_model);
    }

    if(nullptr != m_detect_handle){
        cv_common_detection_hunter_destroy(m_detect_handle);
    }

    if(nullptr != m_detect_model){
        cv_common_unload_model(m_detect_model);
    }
}

int LivenessDetector::init(const char* licenseFile,
                           const char* detectModelFile,
                           const char* alignModelFile,
                           const char* selectorModelFile,
                           const char* hacknessModelFile){
    if(0!= AddLicense(licenseFile))
        return -1;
//    cv_result_t st_result = CV_OK;
    //===========================use hunter==============================

    cv_result_t st_result = cv_common_load_model(detectModelFile, &m_detect_model);

//    model_guard_t detect_model_guard(m_detect_model, cv_common_unload_model);
    if (st_result != CV_OK) {
        LOGE("fail to load hunter model:%d, [%s]", st_result, detectModelFile);
        return -1;
    }

    st_result = cv_common_detection_hunter_create(m_detect_model, &m_detect_handle);
//    handle_guard_t detect_handle_guard(m_detect_handle, cv_common_detection_hunter_destroy);

    if (st_result != CV_OK) {
        LOGE("fail to init detect_hunter handle:%d", st_result);
        return -1;
    }

    //===========================align model===============================

    st_result = cv_common_load_model(alignModelFile, &m_align_model);
//    model_guard_t align_model_guard(m_align_model, cv_common_unload_model);
    if (st_result != CV_OK) {
        LOGE("fail to align hunter model:%d, [%s]", st_result, alignModelFile);
        return -1;
    }

    //===============================tracker =================================
    st_result = cv_common_tracking_compact_create(m_align_model, m_detect_handle, 0,
                                                  &m_track_handle);

//    handle_guard_t track_handle_guard(m_track_handle, cv_common_tracking_compact_destroy);

    if (st_result != CV_OK) {
        LOGE("fail to init track handle:%d\n", st_result);
        return -1;
    }

    st_result = cv_common_tracking_compact_config(
            m_track_handle, CV_COMMON_TRACKING_CONF_DETECTINTERVAL, 20, nullptr);
    if (st_result != CV_OK) {
        LOGE("fail to init track handle config:%d\n", st_result);
        return -1;
    }

    //===========================live selector model==============================

    st_result = cv_common_load_model(selectorModelFile, &m_selector_model);

//    model_guard_t selector_model_guard(m_selector_model, cv_common_unload_model);
    if (st_result != CV_OK) {
        LOGE("fail to selector model:%d, [%s]", st_result,selectorModelFile);
        return -1;
    }

    // init live front handle
    st_result = cv_liveness_verification_frame_selector_106_create(m_selector_model, &m_selector_handle);
//    handle_guard_t selector_handle_guard(m_selector_handle, cv_liveness_verification_frame_selector_destroy);
    if (st_result != CV_OK) {
        LOGE("fail to init selector_handle:%d\n", st_result);
        return -1;
    }
    // reset selector handle
    st_result = cv_liveness_verification_frame_selector_reset(m_selector_handle);

    if (st_result != CV_OK) {
        LOGE("fail to reset selector_handle:%d\n", st_result);
        return -1;
    }

    //===============================hackness =================================

    st_result = cv_common_load_model(hacknessModelFile, &m_hackness_model);
//    model_guard_t hackness_model_guard(m_hackness_model, cv_common_unload_model);
    if (st_result != CV_OK) {
        LOGE("fail to load hackness model: %d, [%s]", st_result,hacknessModelFile);
        return -1;
    }
    st_result = cv_liveness_antispoofing_general_create(m_hackness_model, &m_hackness_handle);
//    handle_guard_t hackness_handle_guard(m_hackness_handle, cv_liveness_antispoofing_general_destroy);
    if (st_result != CV_OK) {
        LOGE("fail to init hackness handle: %d\n", st_result);
        return -1;
    }
    m_best_selector_score = 0.0;
    return 0;
}

namespace {

    void cleanImg(cv_image_t *pImg){
        if(0!=pImg) cv_image_release(pImg);
    }

    int convertImage(const unsigned char *src, int src_length, cv_pixel_format src_format,
                unsigned char *dst, int dst_length, cv_pixel_format dst_format, int width, int height) {

        cv_image_t *src_image=0, *dst_image=0;

        cv_result_t st_result = cv_image_allocate(width, height, src_format, &src_image);
        if (st_result != CV_OK) {
            LOGE("create image failed!");
            return -1;
        }

        st_result = cv_image_allocate(width, height, dst_format, &dst_image);
        if (st_result != CV_OK) {
            LOGE("create image failed!");
            cleanImg(src_image);
            return -1;
        }


        std::copy(src, src + src_length, src_image->data);

        st_result = cv_common_color_convert(src_image, dst_image);

        if (st_result == CV_OK) {
            std::copy(dst_image->data, dst_image->data + dst_length, dst);
        }
        else{
            LOGE("color convert failed!");
        }

        cleanImg(src_image);
        cleanImg(dst_image);
        return (st_result == CV_OK)?0:-1;
    }

    int rotateImage(const unsigned char *src, int length, cv_pixel_format format, unsigned char *dst, int width,
                            int height, unsigned int degree) {

        cv_image_t *src_image=0, *dst_image=0;

        const bool swap = degree == 90 || degree == 270;
        const int dst_width = swap ? height : width;
        const int dst_height = swap ? width : height;

        cv_result_t st_result = cv_image_allocate(width, height, format, &src_image);
        if (st_result != CV_OK) {
            LOGE("create image failed!");
            return -1;
        }
        st_result = cv_image_allocate(dst_width, dst_height, format, &dst_image);
        if (st_result != CV_OK) {
            LOGE("create image failed!");
            cleanImg(src_image);
            return -1;
        }

        std::copy(src, src + length, src_image->data);

        st_result = cv_common_image_rotate(src_image, dst_image, degree);

        if (st_result == CV_OK) {
            std::copy(dst_image->data, dst_image->data + length, dst);
        }
        else{
            LOGE("rotate failed!");
        }
        cv_image_release(src_image);
        cv_image_release(dst_image);
        return (st_result == CV_OK)?0:-1;
    }

}

int LivenessDetector::detectImage(unsigned char *nv21Data, int format, int width, int height, int rotateDegree){

    if (format != CV_PIX_FMT_NV21) {
        LOGE("unsupport format [%d]", format);
        return -1;
    }

    if(nullptr==m_detect_model || nullptr==m_detect_handle || nullptr==m_align_model|| nullptr==m_track_handle||
       nullptr==m_selector_model|| nullptr==m_selector_handle || nullptr==m_hackness_model|| nullptr==m_hackness_handle){
        LOGE("null model/handle");
        return -1;
    }

    LOGI("image w[%d], h[%d], degree[%d]", width, height, rotateDegree);

    int ret = 0;

    cv_time_t current_time = {0,0};

    cv_image_t nvImg = {
            nv21Data,
            CV_PIX_FMT_NV21,
            width,
            height,
            width,
            current_time
    };

    cv_image_t *colorImg = 0;
//    cv_result_t st_result= cv_image_allocate(width, height, CV_PIX_FMT_BGR888, &colorImg);
    cv_result_t st_result= cv_image_allocate(width, height, CV_PIX_FMT_RGB888, &colorImg);
    if (st_result != CV_OK) {
        LOGE("create image failed: %d", st_result);
        return -1;
    }
    st_result = cv_common_color_convert(&nvImg, colorImg);
    if (st_result != CV_OK) {
        LOGE("color convert failed: %d", st_result);
        cv_image_release(colorImg);
        return -1;
    }

//    run(*colorImg, rotateDegree);
    run(nvImg, rotateDegree);
    cv_image_release(colorImg);
    return 0;
}



//int LivenessDetector::detectImage(const unsigned char *nv21Data, int format, int width, int height, int rotateDegree){
//
//    if (format != CV_PIX_FMT_NV21) {
//        LOGE("unsupport format [%d]", format);
//        return -1;
//    }
//
//    if(nullptr==m_detect_model || nullptr==m_detect_handle || nullptr==m_align_model|| nullptr==m_track_handle||
//        nullptr==m_selector_model|| nullptr==m_selector_handle || nullptr==m_hackness_model|| nullptr==m_hackness_handle){
//        LOGE("null model/handle");
//        return -1;
//    }
//
//    LOGI("image w[%d], h[%d], degree[%d]", width, height, rotateDegree);
//
//    int ret = 0;
//
//    if(false)
//    {
//        int nv21_data_length = width * height * 3 / 2;
//
//        unsigned char *nv21_rotated_data = new unsigned char[nv21_data_length];
//
//        if(90 == rotateDegree || 180 == rotateDegree || 270 == rotateDegree){
//            ret = rotateImage(nv21Data, nv21_data_length, CV_PIX_FMT_NV21, nv21_rotated_data, width, height, rotateDegree);
//
//            if(0!= ret){
//                delete[] nv21_rotated_data;
//                return -1;
//            }
//        } else if(0==rotateDegree){
//            std::copy(nv21Data, nv21Data + nv21_data_length, nv21_rotated_data);
//        } else{
//            LOGE("wrong degree[%d]", rotateDegree);
//            return -1;
//        }
//
//        if(90 == rotateDegree || 270 == rotateDegree)
//            std::swap(width, height);
//
//        LOGI("after rotate w[%d], h[%d]", width, height);
//
//        unsigned char *rgbData = new unsigned char[width * height * 3];
//        ret = convertImage(nv21_rotated_data, width * height * 3 / 2, CV_PIX_FMT_NV21, rgbData, width * height * 3,
//                           CV_PIX_FMT_BGR888, width, height);
//        delete[] nv21_rotated_data;
//        if(0!= ret){
//            delete[] rgbData;
//            return -1;
//        }
//    }
//
//
//    unsigned char *colorData = new unsigned char[width * height * 3];
//    ret = convertImage(nv21Data, width * height * 3 / 2, CV_PIX_FMT_NV21, colorData, width * height * 3,
//                       CV_PIX_FMT_BGR888, width, height);
//    if(0!= ret){
//        delete[] colorData;
//        return -1;
//    }
//
//    unsigned char *grayData = new unsigned char[width * height];
//    ret = convertImage(nv21Data, width * height * 3 / 2, CV_PIX_FMT_NV21, grayData, width * height,
//                       CV_PIX_FMT_GRAY8, width, height);
//    if(0!= ret){
//        delete[] grayData;
//        return -1;
//    }
//
//
//    cv_time_t current_time = {0,0};
//
//    cv_image_t colorImg = {
//            colorData,
//            CV_PIX_FMT_BGR888,
//            width,
//            height,
//            width*3,
//            current_time
//    };
//
//    cv_image_t grayImg = {
//            colorData,
//            CV_PIX_FMT_GRAY8,
//            width,
//            height,
//            width,
//            current_time
//    };
//
//
//
////    {
////        cv_image_t* img_gray = 0;
////        cv_result_t st_result = cv_image_allocate(img_rgb.width, img_rgb.height, CV_PIX_FMT_GRAY8, &img_gray);
////        if (st_result != CV_OK) {
////            LOGE("create gray image failed: %d", st_result);
////            return -1;
////        }
////
////        st_result = cv_common_color_convert(&img_rgb, img_gray);
////        if (st_result != CV_OK) {
////            LOGE("rgb2gray failed: %d", st_result);
////            cleanImg(img_gray);
////            return -1;
////        }
////        delete[] rgbData;
////
////        cv_image_t *input_img = rotateImage(*img_gray,rotateDegree);
////        run(*input_img);
////        cv_image_release(input_img);
////    }
//
//
//
////    cv_image_t *tmp_img = nullptr;
////    cv_image_allocate(width, height, CV_PIX_FMT_BGR888, &tmp_img);
////    memcpy(tmp_img->data, rgbData, width * height * 3);
//
//
////    char colorFile[80]={0};
////    char grayFile[80]={0};
////
////    sprintf(colorFile, k_color_file, m_imageIndex);
////    sprintf(grayFile, k_gray_file, m_imageIndex);
////
////
////    if(m_imageIndex<5)
////    {
////        Mat c1(colorImg.height, colorImg.width, IMREAD_COLOR, colorImg.data);
////        imwrite(colorFile, c1);
////
////        Mat r1(grayImg.height, grayImg.width, IMREAD_GRAYSCALE, grayImg.data);
////        imwrite(grayFile, r1);
////        m_imageIndex++;
////        sleep(1);
////    }
//
////    delete[] rgbData;
////
//////    cv_image_release(tmp_img);
////
////    delete[] rgbData;
////
////
////    if(nullptr == input_img){
////        LOGE("rotate image failed!");
////        return -1;
////    }
////
////
//
////    run(colorImg);
//    run(colorImg, rotateDegree);
//    delete[] colorData;
//    delete[] grayData;
//    return 0;
//}

int LivenessDetector::run(const cv_image_t & inputImage, int degree){
    cv_image_t* img_gray = 0;
    cv_result_t st_result = cv_image_allocate(inputImage.width, inputImage.height, CV_PIX_FMT_GRAY8, &img_gray);
    if (st_result != CV_OK) {
        LOGE("create gray image failed: %d", st_result);
        return -1;
    }

    st_result = cv_common_color_convert(&inputImage, img_gray);
    if (st_result != CV_OK) {
        LOGE("convert gray failed: %d", st_result);
        cleanImg(img_gray);
        return -1;
    }

    cv_target_t *faces = nullptr;
    int faces_count = 0;

    LOGI("begin tracking_compact_track!");
    cv_orientation orientation = CV_ORIENTATION_0;

    if(90 == degree)
        orientation=CV_ORIENTATION_90;
    else if(180 == degree)
        orientation=CV_ORIENTATION_180;
    else if(270 == degree)
        orientation=CV_ORIENTATION_270;

    st_result = cv_common_tracking_compact_track(
            m_track_handle, img_gray, orientation, &faces, &faces_count);
    if (st_result != CV_OK) {
        LOGE("cv_common_tracking_compact_track failed: %d", st_result);
        cleanImg(img_gray);
        return -1;
    }

    cv_landmarks_t landmarks;
    float selector_score = 0.0f;

    LOGI("face count[%d]!", faces_count);

    if (faces_count > 0) {
//        for (size_t i_point = 0; i_point < 106; ++i_point) {
//            float x = faces[0].landmarks.points_array[i_point].x;
//            float y = faces[0].landmarks.points_array[i_point].y;
//            circle(image_color, Point2f(x, y), 2, Scalar(0, 0, 255), -1);
//        }
        landmarks.points_array = faces[0].landmarks.points_array;
        LOGI("points count[%d]!", faces[0].landmarks.points_count);
        landmarks.points_count = 106;

        LOGI("begin get_score!");

        st_result = cv_liveness_verification_frame_selector_get_score(
                m_selector_handle, img_gray, &landmarks, &selector_score);

        if (st_result != CV_OK) {
            LOGE("cv_liveness_verification_frame_selector_get_score failed: %d\n", st_result);
            cleanImg(img_gray);
            return -1;
        }
    }

    LOGI("current score: [%.2f], best score: [%.2f]", selector_score, m_best_selector_score);

    if (selector_score > m_best_selector_score) {
        m_best_selector_score = selector_score;
        float score_hackness = 0;
        LOGI("begin antispoofing!");
        st_result = cv_liveness_antispoofing_general_get_score(m_hackness_handle, &inputImage, &landmarks, &score_hackness);

        if (st_result != CV_OK) {
            LOGE("run cv_liveness_antispoofing_general_get_score error: %d\n", st_result);
            cleanImg(img_gray);
            return -1;
        }
        LOGI("hackness score: %f\n", score_hackness);
    }

    cleanImg(img_gray);
    return 0;
}

void LivenessDetector::reset(){
    m_best_selector_score=0;
}

int LivenessDetector::detectImageFromFile(const char* imageFile)
{
    LOGI("image [%s]", imageFile);
    cv::Mat img = cv::imread(imageFile);
    LOGI("image data[%p], col[%d], row[%d], channel[%d], step[%d]", img.data, img.cols, img.rows,img.channels(),(int)img.step);

//    Mat rotate= Mat(img.rows, img.cols,img.depth());
//    transpose(gray, rotate);
//    flip(gray, rotate, 1);  //rotate 270
//    flip(gray, rotate, 0);  //rotate 90

    cv_image_t input_image = {img.data, CV_PIX_FMT_BGR888,
                              img.cols, img.rows,
                              static_cast<int>(img.step), {0,0}};
    run(input_image,270);
    return 0;
}

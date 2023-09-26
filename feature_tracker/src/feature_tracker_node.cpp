#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

//特征提取的主入口，负责一个特征处理结点的功能

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

// 该函数是ROS的回调函数，主要功能包括：readImage()函数对新来的图像使用光流法进行特征点跟踪，
// 并将追踪的特征点封装成feature_points发布到pub_img的话题下，将图像封装成ptr发布在pub_match下。
// 数据结构：
//    *init_feature         0:当前是第一帧   1:当前不是第一帧
//    *first_image_flag     0:当前是第一帧   1:当前不是第一帧
//    *pub_count            每隔delta_t = 1/FREQ 时间内连续(没有中断/没有报错)发布的帧数
//    *first_image_time     每隔delta_t = 1/FREQ 时间的帧对应的时间戳 
//    *last_image_time      当前帧或上一帧的时间戳 
//    *restart_flag         0:不重启。       1:重启(std_msgs::Bool)
//    *FREQ                 发布特征点的频率(也就是bk/ck帧与bk+1/ck+1帧之间的频率)，注意，FREQ要比实际的照片频率要慢
//    *init_pub             0:第一帧不把特征发布到buf里    1:发布到buf里    
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag) //对于第一帧图像，只记录对应时间戳，不提取特征，因为他没有前一帧图像，无法获取光流。
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream 
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time) //对于时间戳错乱的帧，重新初始化
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    
    // 2. frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ) //要想发布图像帧，那么实际频率要比设定值小
    {
        PUB_THIS_FRAME = true; //发布特征点
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)// 处理累计误差，如果实际频率与设定频率的累积误差大于0.01了，就不能发布这一帧
        {
            first_image_time = img_msg->header.stamp.toSec(); 
            pub_count = 0; //pub_count最大只能是1，因为只要能发布一帧，这个数就会被清0
        }
    }
    else   //如果实际发布频率大于设定值，肯定就不发了
        PUB_THIS_FRAME = false;

    //3. 图像的格式调整和图像读取
    //读取sensor_msgs::Image img的数据，并转为MONO8格式，用cv::Mat show_img接收
    //构建了CV:Mat与sensor_masg::Image之间的桥梁。
    //注意，img_msg或img都是sensor_msg格式的，我们需要一个桥梁，转换为CV::Mat格式的数据，以供后续图像处理。
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") //8位1通道，即：灰度图
    {
        sensor_msgs::Image img;  //估计是为了将const的img_msg转换为非const的img
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8); ////将图像编码8UC1转换为mono8,即存储下来的图像为单色，8Bit的图片，一般是bmp，jpeg等
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    
    cv::Mat show_img = ptr->image;
    TicToc t_r;

    //对最新帧forw的特征点的提取和光流追踪(核心)
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK) //单目时：FeatureTracker::readImage() 函数读取图像数据进行处理
            //readImage()这个函数实现了特征的处理和光流的追踪，里面基本上调用了feature_tracker.cpp里面的全部函数
            //readImage()传了2个参数，当前帧的图像和当前帧的时间戳
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE) //判断是否对图像进行自适应直方图均衡化
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    // 对新加入的特征点更新全局id
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed) // completed(或者是update())如果是true，说明没有更新完id，则持续循环，如果是false，说明更新完了则跳出循环
            break; 
    }

    //6. 特征点的发布
   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud); //归一化坐标
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point; //像素坐标x
        sensor_msgs::ChannelFloat32 v_of_point; //像素坐标y
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts; //归一化坐标
            auto &cur_pts = trackerData[i].cur_pts;   //像素坐标
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)  //只发布追踪次数大于1的特征点
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;  //归一化坐标
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);  //归一化坐标
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x); //像素坐标
                    v_of_point.values.push_back(cur_pts[j].y); //像素坐标
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            } //将特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)，封装成sensor_msgs::PointCloudPtr类型的feature_points实例中,发布到pub_img
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        
        // /如果是第一帧的话，不发布数据
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        // 7. 将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");//ros初始化
    ros::NodeHandle n("~"); //设置句柄
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info); //设置logger级别
    readParameters(n); //读取如 config->euroc->euroc_config.yaml 中的一些配置参数；

    for (int i = 0; i < NUM_OF_CAM; i++) //读取每个相机实例对应的相机内参，NUM_OF_CAM=1为单目
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE) //判断是否加入鱼眼mask来去除边缘噪声
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback); //订阅话题IMAGE_TOPIC(如/cam0/image_raw)，执行回调函数img_callback
                                                                           // IMAGE_TOPIC就是配置文件中定义的/cam0/image_raw，
                                                                           //一旦订阅到图像信息，那么就执行回调函数img_callback。

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000); //跟踪的特征点图像，主要是之后给RVIZ用和调试用
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000); //即跟踪的特征点信息，由/vins_estimator订阅并进行优化
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);       //判断特征跟踪模块是否出错，若有问题则进行复位，由/vins_estimator订阅
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?
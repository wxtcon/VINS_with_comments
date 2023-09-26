#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// 预积分 第一次出现
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec(); //获取当前时间
    if (init_imu) //首帧判断
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time; //获取dt并传递时间
    latest_time = t;

    //获取当前时刻的IMU采样数据
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //注意，以下数据都是世界坐标系下的
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    //信息传递
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update() //这个函数在非线性优化时才会在process()中被调用
{
    //1、从估计器中得到滑动窗口中最后一个图像帧的imu更新项[P,Q,V,ba,bg,a,g]
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    //2、对imu_buf中剩余的imu_msg进行PVQ递推（因为imu的频率比图像频率要高很多，在getMeasurements(）将图像和imu时间对齐后，imu_buf中还会存在imu数据）
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

//这个函数的作用就是 把图像帧 和 对应的IMU数据们 配对起来,而且IMU数据时间是在图像帧的前面
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //边界判断：数据取完了，说明配对完成
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //边界判断：IMU buf里面所有数据的时间戳都比img buf第一个帧时间戳要早，说明缺乏IMU数据，需要等待IMU数据
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;  //统计等待的次数
            return measurements;
        }

        //边界判断：IMU第一个数据的时间要大于第一个图像特征数据的时间(说明图像帧有多的)
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        
        //核心操作：装入视觉帧信息
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        //核心操作：转入IMU信息
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        //注意：把最后一个IMU帧又放回到imu_buf里去了
        //原因：最后一帧IMU信息是被相邻2个视觉帧共享的
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// imu_callback主要干了3件事，
// 第一件事就是往imu_buf里放IMU数据，缓存起来；
// 第二件事就是IMU预积分获得当前时刻的PVQ，
// 第三件事就是如果当前处于非线性优化阶段的话，需要把第二件事计算得到的PVQ发布到rviz里去，见utility/visualization.cpp的pubLatestOdometry()函数。
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock(); //对imu_buf的操作是一个生产者-消费者模型，加入和读取的时候不可以中断，必须加锁以保证数据安全
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); //tmp_Q，tmp_P，tmp_V：当前时刻的PVQ     header：当前时刻时间戳 
    }
}


// feature_callback就只干了一件事，就是把cur帧的所有特征点放到feature_buf里，同样需要上锁。
// cur帧的所有特征点都是整合在一个数据里的，也就是 sensor_msgs::PointCloudConstPtr &feature_msg
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// restart_callback 干了一件事，就是把所有状态量归零，把buf里的数据全部清空。
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        // 1、对imu和图像数据进行对齐并配对 作用就是 把图像帧和对应的IMU数据们 配对起来,而且IMU数据时间是在图像帧的前面
        
        // 数据结构: measurements
        // 1、首先，measurements他自己就是一个vector；
        // 2、对于measurements中的每一个measurement，又由2部分组成；
        // 3、第一部分，由sensor_msgs::ImuConstPtr组成的vector；
        // 4、第二部分，一个sensor_msgs::PointCloudConstPtr；
        // 5、这两个sensor_msgs见3.1-6部分介绍。
        // 6、为什么要这样配对(一个PointCloudConstPtr配上若干个ImuConstPtr)？
        // 因为IMU的频率比视觉帧的发布频率要高，所以说在这里，需要把一个视觉帧和之前的一串IMU帧的数据配对起来。 
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements) //2、对measurements中的每一个measurement (IMUs,IMG)组合进行操作
        {
            auto img_msg = measurement.second; //2.1、对于measurement中的每一个imu_msg，计算dt并执行processIMU()
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td; //相机和IMU同步校准得到的时间差
                
                //对于大多数情况，IMU的时间戳都会比img的早，此时直接选取IMU的数据就行
                if (t <= img_t)  //这个if-else的操作技巧非常值得学习！这一部分的核心代码是processIMU()，它在estomator.cpp里面，它的作用就是IMU预积分，
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;

                    //这里干了2件事，IMU粗略地预积分，然后把值传给一个新建的IntegrationBase对象
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                //对于处于边界位置的IMU数据，是被相邻两帧共享的，而且对前一帧的影响会大一些，在这里，对数据线性分配
                //每个大于图像帧时间戳的第一个imu_msg是被两个图像帧共用的(出现次数少)
                else 
                {
                    double dt_1 = img_t - current_time; //current_time < img_time < t
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    
                    //以下操作其实就是简单的线性分配
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz)); //IMU预积分
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            // 下面代码的作用是，在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame()
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            //2.2、在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame()
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r); //设置重定位帧
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            //4、对img信息进行处理(核心！)
            TicToc t_s;
            // 数据结构: map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image
            // 1、虽然它叫image，但是这个容器里面存放的信息是每一个特征点的！
            // 2、索引值是feature_id；
            // 3、value值是一个vector，如果系统是多目的，那么同一个特征点在不同摄像头下会有不同的观测信息，那么这个vector，就是存储着某个特征点在所有摄像头上的信息。
            //    对于VINS-mono来说，value它不是vector，仅仅是一个pair，其实是可以的。
            // 4、接下来看这个vector里面的每一pair。int对应的是camera_id，告诉我们这些数据是当前特征点在哪个摄像头上获得的。
            // 5、Matrix<double, 7, 1>是一个7维向量，依次存放着当前feature_id的特征点在camera_id的相机中的归一化坐标，像素坐标和像素运动速度，
            //    这些信息都是在feature_tracker_node.cpp中获得的。   
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++) //遍历img_msg里面的每一个特征点的归一化坐标
            {
                //把img的信息提取出来放在image容器里去，通过这里，可以理解img信息里面装的都是些什么
                int v = img_msg->channels[0].values[i] + 0.5; //channels[0].values[i]==id_of_point
                //hash
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            //(3) 处理图像processImage() (核心！)
            //这里实现了视觉与IMU的初始化以及非线性优化的紧耦合。这一部分的内容非常多
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            // 5.可视化
            // 向RVIZ发布里程计信息、关键位姿、相机位姿、点云和TF关系，这些函数都定义在中utility/visualization.cpp里，都是ROS相关代码。
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        // 6. IMU的PVQ信息更新
        // 更新IMU参数[P,Q,V,ba,bg,a,g]，需要上锁，注意线程安全。
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        

        // 因为在进行完一个process()循环后，当前的PVQ的状态和循环开始的状态是不一样的。
        // 所以说我们需要再根据当前的数据，更新当前的PVQ状态，也就是tmp_X。同样，得上锁。
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

//vins_estimator入口函数
int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n); //读取参数
    estimator.setParameter(); //设置状态估计器参数
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n); //发布用于RVIZ显示的Topic 这个函数定义在utility/visualization.cpp里面：void registerPub(ros::NodeHandle &n)。

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay()); //订阅IMU_TOPIC，执行 imu_callback
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback); //订阅 /feature_tracker/feature，执行 feature_callback
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback); //订阅/feature_tracker/restart，执行 restart_callback
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback); // 订阅/pose_graph/match_points，执行relocalization_callback

    std::thread measurement_process{process}; //创建VIO主线程process()(VINS核心！) 这一部分是最重要的，包含了VINS绝大部分内容
    ros::spin();

    return 0;
}

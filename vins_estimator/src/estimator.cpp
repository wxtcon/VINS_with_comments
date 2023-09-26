#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

// 它读取了每一个相机到IMU坐标系的旋转/平移外参数和非线性优化的重投影误差部分的信息矩阵。
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}


// 数据结构:
// 1、Rs[frame_count]，Ps[frame_count]，Vs[frame_count]:是从IMU系转到world系的PVQ，数据是由IMU预积分得到的，
//    目前在这里存放的是没有用bias修正过的值。
// 2、frame_count:这个值让我很疑惑，它只在processImage()里有过++操作，而且在estimator.hpp声明的时候，没有加上static关键字。
//    它是在h文件中声明，在cpp文件里初始化的，后续需要再关注一下。
// 3、dt_buf，linear_acceleration_buf，angular_velocity_buf：帧数和IMU测量值的缓存，而且它们是对齐的。
// 3、pre_integrations[frame_count]，它是IntegrationBase的一个实例，在factor/integration_base.h中定义，
//    它保存着frame_count帧中所有跟IMU预积分相关的量，包括F矩阵，Q矩阵，J矩阵等。    
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    /*这一段作用就是就是给以下数据提供初始值/初始化：
   * 
   *pre_integrations[frame_count]
   *dt_buf[frame_count]
   *linear_acceleration_buf[frame_count]
   *angular_velocity_buf[frame_count]
   *Rs[frame_count]
   *PS[frame_count]
   *Vs[frame_count] 
   * 
   * TODO 关于frame_count的更新，目前只在process_img里的solver_flag == INITIAL这里看到?
   * 
   */

    //边界判断：如果当前帧不是第一帧IMU，那么就把它看成第一个IMU，而且把他的值取出来作为初始值
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    //边界判断：如果当前IMU帧没有构造IntegrationBase，那就构造一个，后续会用上
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    
    //核心操作
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        
        //注意啊，这块对j的操作看似反智，是因为j时刻的值都拷贝了j-1时刻的值！！
        //第一次使用实际上就是使用的是j-1时刻的值，所以在这些地方写上j-1是没有关系的！
        //noise是zero mean Gauss，在这里忽略了
        //TODO 把j改成j-1，看看效果是一样
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        //下面都采用的是中值积分的传播方式，noise被忽略了
        //TODO 把j改成j-1，看看效果是一样
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }

    //数据传递
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 它传入的参数分别是当前帧上的所有特征点和当前帧的时间戳
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    
    //VINS滑动窗口采取的是这样的策略，它判断当前帧是不是关键帧，如果是关键帧，滑窗的时候marg掉最老帧；如果不是关键帧，则marg掉上一帧。
    //true：上一帧是关键帧，marg_old; 
    //false:上一帧不是关键帧 marg_second_new
    
    //数据结构: f_manager
    //    f_manager是FeatureManager的一个对象。
    //    它定义在utility/feature_manager.h里。这个h文件里定义了3个类，
    // f_manager可以看作为一个存放着滑窗内所有特征点信息的容器，其中最关键的部分是list<FeaturePerId> feature。
    // 其中每一个特征点，可以看作是一个FeaturePerId的对象，它存放着一个特征点在滑窗中的所有信息，其中最关键的部分是vector<FeaturePerFrame> feature_per_frame。
    // 其中一个特征点在一个帧中的信息，可以看作是一个FeaturePerFrame的对象，它存放着一个特征点在滑窗里一个帧里面的信息，包括归一化坐标，像素坐标，像素速度等。
    //套娃套了三层。
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // 2、将图像数据、时间、临时预积分值存到图像帧类中
    // 数据结构: ImageFrame imageframe
    // imageframe是ImageFrame的一个实例，定义在initial/initial_alignment.h里。
    // 顾名思义，它是用于融合IMU和视觉信息的数据结构，包括了某一帧的全部信息:位姿，特征点信息，预积分信息，是否是关键帧等。
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    // 3、更新临时预积分初始值
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 4、如果需要标定外参，则标定
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 5、初始化 一般初始化只进行一次。
    // 初始化部分的代码虽然生命周期比较短，但是，代码量巨大！
    // 主要分成2部分，第一部分是纯视觉SfM优化滑窗内的位姿，然后再融合IMU信息，按照理论部分优化各个状态量。
    if (solver_flag == INITIAL) //进行初始化
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            //确保有足够的frame参与初始化，有外参，且当前帧时间戳大于初始化时间戳+0.1秒
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure(); //执行视觉惯性联合初始化，套娃函数，具体看笔记
               initial_timestamp = header.stamp.toSec(); //更新初始化时间戳
            }
            if(result) //初始化成功则进行一次非线性优化
            {
                solver_flag = NON_LINEAR; //进行非线性优化
                solveOdometry(); //执行非线性优化具体函数solveOdometry()
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];  //得到当前帧与第一帧的位姿
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow(); //不成功则进行滑窗操作
        }
        else
            frame_count++; //图像帧数量+1
    }
    else  //此时，solver_flag = NON_LINEAR,进行非线性优化，在主流程中，可以发现干了6件事，
          //其中1，3是最关键的2个步骤。而optimization()恰好在solveOdometry()里，所以可以发现，代码是先优化，后marg。
    {
        TicToc t_solve;
        solveOdometry(); //<--1
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        //边界判断：检测系统运行是否失败，若失败则重置估计器
        if (failureDetection()) //<--2
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow(); //执行窗口滑动函数slideWindow();//<--3
        f_manager.removeFailures(); //去除估计失败的点并发布关键点位置//<--4
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear(); //<--5
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE]; //<--6
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    // 确保IMU有足够的激励
    // 这一部分的思想就是通过计算滑窗内所有帧的线加速度的标准差，判断IMU是否有充分运动激励，以判断是否进行初始化
    // all_image_frame这个数据结构，见5.2-2，它包含了滑窗内所有帧的视觉和IMU信息，它是一个hash，以时间戳为索引。
    {
        map<double, ImageFrame>::iterator frame_it;
        // 1. 第一次循环，求出滑窗内的平均线加速度
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt; //time for bk to bk+1
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        
        // 2. 第二次循环，求出滑窗内的线加速度的标准差
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g); //计算加速度的方差
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); //计算加速度的标准差
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    // 将f_manager中的所有feature在所有帧的归一化坐标保存到vector sfm_f中(辅助)
    Quaterniond Q[frame_count + 1]; // 为什么容量是frame_count + 1？因为滑窗的容量是10，再加上当前最新帧，所以需要储存11帧的值！
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    // 数据结构: vector<SFMFeature> sfm_f
    // 它定义在initial/initial_sfm.h中， 可以发现，它存放着一个特征点的所有信息。容器定义完了，接下来就是往容器里放数据。
    //     struct SFMFeature
    // {
    //     bool state;//状态（是否被三角化）
    //     int id;
    //     vector<pair<int,Vector2d>> observation;//所有观测到该特征点的 图像帧ID 和 特征点在这个图像帧的归一化坐标
    //     double position[3];//在帧l下的空间坐标
    //     double depth;//深度
    // };   
    // 在这里，为什么要多此一举构造一个sfm_f而不是直接使用f_manager呢？
    // 我的理解，是因为f_manager的信息量大于SfM所需的信息量(f_manager包含了大量的像素信息)，
    // 而且不同的数据结构是为了不同的业务服务的，所以在这里作者专门为SfM设计了一个全新的数据结构sfm_f，专业化服务。
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature) //对于滑窗中出现的 所有特征点
    {
        int imu_j = it_per_id.start_frame - 1; //从start_frame开始帧编号
        SFMFeature tmp_feature;
        tmp_feature.state = false; //状态（是否被三角化）
        tmp_feature.id = it_per_id.feature_id; //特征点id
        for (auto &it_per_frame : it_per_id.feature_per_frame) //对于当前特征点 在每一帧的坐标
        {
            imu_j++; //帧编号+1
            Vector3d pts_j = it_per_frame.point; //当前特征在编号为imu_j++的帧的归一化坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));//把当前特征在当前帧的坐标和当前帧的编号配上对
        } //tmp_feature.observation里面存放着的一直是同一个特征，每一个pair是这个特征在 不同帧号 中的 归一化坐标
        sfm_f.push_back(tmp_feature); //sfm_f里面存放着是不同特征
    } 
    
    // 3. 在滑窗(0-9)中找到第一个满足要求的帧(第l帧)，它与最新一帧(frame_count=10)有足够的共视点和平行度，并求出这两帧之间的相对位置变化关系
    // 定义容器
    Matrix3d relative_R;
    Vector3d relative_T;
    int l; //滑窗中满足与最新帧视差关系的那一帧的帧号
    // 两帧之间的视差判断,并得到两帧之间的相对位姿变化关系
    //这里的第L帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧l，会作为 参考帧 到下面的全局sfm使用，得到的Rt为当前帧到第l帧的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l)) //计算滑窗内的每一帧(0-9)与最新一帧(10)之间的视差，直到找出第一个满足要求的帧，作为我们的第l帧；
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    GlobalSFM sfm;
    // 这里主要干了2件事
    // 首先SfM，传入了frame_count + 1，l, relative_R, relative_T, sfm_f这几个参数，得到了Q, T, sfm_tracked_points，这三个量的都是基于l帧上表示的！
    // 第二件事就是marginalization_flag = MARGIN_OLD。这说明了在初始化后期的第一次slidingwindow() marg掉的是old帧。
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    // 给滑窗外的图像帧提供初始的RT估计，然后solvePnP进行优化
    // 目的：求出所有帧对应的IMU坐标系bk到l帧的旋转和ck到l帧的平移，和论文中初始化部分头两个公式对应上！
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    // ImageFrame这个数据结构不仅包括了图像信息，还包括了对应的IMU的位姿信息和IMU预积分信息,
    // 而这里，是这些帧第一次获得它们对应的IMU的位姿信息的位置！也就是bk->l帧的旋转平移！
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++) // 遍历所有的图像帧
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        //对当前图像帧的操作
        // a.边界判断：对于滑窗内的帧，把它们设为关键帧，并获得它们对应的IMU坐标系到l系的旋转平移
        // 要注意一下，Headers，Q和T，它们的size都是WINDOW_SIZE+1！它们存储的信息都是滑窗内的，尤其是Q和T，它们都是当前视觉帧到l帧(也是视觉帧)系到旋转平移。
        // 所以一开始，通过时间戳判断是不是滑窗内的帧；
        // 如果是，那么设置为关键帧；

        if((frame_it->first) == Headers[i].stamp.toSec()) 
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
            
        }
        
        // 边界判断：如果当前帧的时间戳大于滑窗内第i帧的时间戳，那么i++
        // 这一部分有点像kmp算法中的2把尺子，一把大一点的尺子是all_image_frame，另一把小尺子是Headers，分别对应着所有帧的长度和滑窗长度；
        // 小尺子固定，大尺子在上面滑动；
        // 每次循环，大尺子滑动一格；
        // 因为小尺子比较靠后，所以开始的时候只有大尺子在动，小尺子不动；
        // 如果大尺子和小尺子刻度一样的时候，小尺子也走一格；
        // 如果大尺子的刻度比小尺子大，小尺子走一格；
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        
        // c.对滑窗外的所有帧，求出它们对应的IMU坐标系到l帧的旋转平移
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix(); //注意这里的 Q和 T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵，两者具有对称关系
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec); //罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector; //获取 pnp需要用到的存储每个特征点三维点和图像坐标的 vector
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            } 
        } //保证特征点数大于 5
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp; // 这块明显是告诉我们传入的是bk->l的旋转平移。
    }
    if (visualInitialAlign()) // 核心, 基本上，初始化的理论部分都在visualInitialAlign()函数里。
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // 计算陀螺仪偏置，尺度，重力加速度和速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // 传递所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // 重新计算所有f_manager的特征点深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1; //将所有特征点的深度置为-1
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic //重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    // 这里有2个ric，小ric是vins自己计算得到的，大RIC是从yaml读取的，我觉得没什么区别。
    // 这里需要注意一下，这个triangulate()和之前出现的并不是同一个函数。
    // 它是定义在feature_manager.cpp里的，之前的是在initial_sfm.cpp里面。它们服务的对象是不同的数据结构！
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    // IMU的bias改变，重新计算滑窗内的预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 将Ps、Vs、depth尺度s缩放后从l帧转变为相对于c0帧图像坐标系下
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;

    // 所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true; //至此，初始化的工作全部完成！！
    // 代码量巨长，一半的工作在于视觉SfM(这部分作用仅仅负责求相机pose)，另一半才是论文里说的松耦合初始化！在initialStructure()的前半部分里，
    // 新建了一组sfm_f这样一批特征点，虽然和f_manager有数据重复，但它们的作用仅仅是用来配合求pose的，而且这个数据结构是建在栈区的，函数结束后统统销毁。
    // 而且找l帧这个操作很巧妙。
    // 后半部分，就是论文里的内容。见初始化(理论部分)。
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++) //滑窗内的所有帧都和最新一帧进行视差比较
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); //寻找第i帧到窗口最后一帧(当前帧)的对应特征点归一化坐标
        if (corres.size() > 20) //归一化坐标point(x,y,不需要z)
        {
            double sum_parallax = 0;
            double average_parallax; //计算平均视差
            for (int j = 0; j < int(corres.size()); j++) //第j个对应点在第i帧和最后一帧的(x,y)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1)); //改成3，4呢(对应像素坐标，1-3是归一化xyz坐标)
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size()); //计算平均视差
            
            // 计算l帧与最新一帧的相对位姿关系
            // 最核心的公式就是m_estimator.solveRelativeRT()，这部分非常地关键。这里面代码很简单，就是把对应的点传进入，然后套cv的公式，但是求出来的R和T是谁转向谁的比较容易迷糊。
            // 这个relative_R和relative_T是把最新一帧旋转到第l帧的旋转平移！
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) //判断是否满足初始化条件：视差>30
            { //solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
                l = i; //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的relative_R，relative_T
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            } //一旦这一帧与当前帧视差足够大了，那么就不再继续找下去了(再找只会和当前帧的视差越来越小）
        }
    }
    return false; // 没有满足要求的帧，整个初始化initialStructure()失败
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric); //第一件事就是根据当前的位姿三角化特征点，获得最新特征的深度//<--1
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization(); //<--2
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    //这块固定了先验信息
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]); 
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        //回环帧在当前帧世界坐标系下的位姿
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        //回环帧在当前帧世界坐标系和他自己世界坐标系下位姿的相对变化，这个会用来矫正滑窗内世界坐标系
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        //回环帧和重定位帧相对的位姿变换
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}


void Estimator::optimization()
{
    // a. 声明和引入鲁棒核函数
    ceres::Problem problem;
    ceres::LossFunction *loss_function; //1.引入鲁棒核函数
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    // b. 添加各种待优化量X——位姿优化量
    for (int i = 0; i < WINDOW_SIZE + 1; i++) //还包括最新的第11帧
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        // 这块出现了新数据结构para_Pose[i]和para_SpeedBias[i]，这是因为ceres传入的都是double类型的，在vector2double()里初始化的。
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // c. 添加各种待优化量X——相机外参
    for (int i = 0; i < NUM_OF_CAM; i++) //  7维、相机IMU外参
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC) //如果IMU-相机外参不需要标定
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]); //这个变量固定为constant
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // d. 添加各种待优化量X——IMU-image时间同步误差
    if (ESTIMATE_TD) //  1维，标定同步时间
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double(); //给ParameterBlock赋值。

    // f. 添加各种残差——先验信残差
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // g. 添加各种残差——IMU残差
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    
    // h. 添加各种残差——重投影残差
    // 这里需要再次注意一点，IMU的残差是相邻两帧，但是视觉不是的
    // 分析一下代码，它加入的2帧，这两帧是观测到同一特征的最近两帧。
    int f_m_cnt = 0; //统计有多少个特征用于非线性优化
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature) //遍历每一个特征
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue; //必须满足出现2次以上且在倒数第二帧之前出现过
 
        ++feature_index; //统计有效特征数量  
        //！得到观测到该特征点的首帧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        //！得到首帧观测到的特征点的归一化相机坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame) //遍历当前特征在每一帧的信息
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point; //！得到第二个特征点
            if (ESTIMATE_TD) //在有同步误差的情况下
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else /在没有同步误差的情况下
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    //h. 添加各种残差——回环检测
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature) //遍历滑窗内的每一个点
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size(); //更新这个点在滑窗内出现的次数
            
            //当前点至少被观测到2次，并且首次观测不晚于倒数第二帧
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            
            //当前这个特征点至少要在重定位帧被观测到
            if(start <= relo_frame_local_index)
            {   
                //如果都扫描到滑窗里第i个特征了，回环帧上还有id序号比i小，那这些肯定在滑窗里是没有匹配点的
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                //如果回环帧某个特征点和滑窗内特征点匹配上了
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    //找到这个特征点在回环帧上的归一化坐标
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    //找到这个特征点在滑窗内首次出现的那一帧上的归一化坐标
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    //构造重投影的损失函数
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    //构造这两帧之间的残差块
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++; //回环帧上的这个特征点被用上了，继续找下一个特征点
                }     
            }
        }

    }

    //i.求解
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();

    // (2)边缘化
    // 第二部分，边缘化。这一部分，只边缘化，不求解，求解留给下一轮优化的第一部分来进行。这部分是非常难懂的地方了。    
    TicToc t_whole_marginalization;
    //1)首先，把上一轮残存的信息加进来：
    // 很明显，我现在要marg了，要构造新的先验H矩阵，那么要把之前的老先验的遗留信息加进来。
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        //! 先验误差会一直保存，而不是只使用一次
        //! 如果上一次边缘化的信息存在
        //! 要边缘化的参数块是 para_Pose[0] para_SpeedBias[0] 以及 para_Feature[feature_index](滑窗内的第feature_index个点的逆深度)
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            { //查询last_marginalization_parameter_blocks中是首帧状态量的序号
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor 构造边缘化的的Factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set); //添加上一次边缘化的参数块, 这一段代码用drop_set把最老帧的先验信息干掉了。

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //2)然后，把这次要marg的IMU信息加进来：
        //很明显，被marg掉的是第0帧信息。
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //3)然后，把这次要marg的视觉信息加进来：
        { //添加视觉的先验，只添加起始帧是旧帧且观测次数大于2的Features
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature) //遍历滑窗内所有的Features
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size(); //该特征点被观测到的次数
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue; //Feature的观测次数不小于2次，且起始帧不属于最后两帧

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0) //只选择被边缘化的帧的Features
                    continue;
                //得到该Feature在起始下的归一化坐标
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j) //不需要起始观测帧
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }// vins把第0帧看到的特征点全都扔了。

        // 4) 将三个ResidualBlockInfo中的参数块综合到marginalization_info中
        // 其中，计算所有ResidualBlock(残差项)的残差和雅克比,parameter_block_data是参数块的容器。
        TicToc t_pre_margin;
        marginalization_info->preMarginalize(); //<--1
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc()); 
        
        TicToc t_margin;
        marginalization_info->marginalize(); //<--2
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift; //<--3
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++) //<--4
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD) //<--5
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift); //<--6

        if (last_marginalization_info) //<--7
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info; //<--8
        last_marginalization_parameter_blocks = parameter_blocks;
        
    } // 至此，marg_old结束。
    else
    { // l.marg_new
        // 如果第二最新帧不是关键帧的话，则把这帧的视觉测量舍弃掉（边缘化）而保留IMU测量值在滑动窗口中。（其他步骤和上一步骤相同）
        // else//如果第二最新帧不是关键帧的话，则把这帧的视觉测量舍弃掉（边缘化）而保留IMU测量值在滑动窗口中。（其他步骤和上一步骤相同
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
} // 至此，optimization()结束。

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        //(1) 保存最老帧信息
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        // (2) 依次把滑窗内信息前移
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            // (3) 把滑窗末尾(10帧)信息给最新一帧(11帧)
            // 注意，在第(2)步中，已经实现了所有信息的前移，此时，最新一帧已经成为了滑窗中的第10帧，这里只是把原先的最新一帧的信息作为下一次最新一帧的初始值。
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            //(4) 新实例化一个IMU预积分对象给下一个最新一帧
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            //(5) 清空第11帧的buf
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0); //6.找到滑窗内最老一帧信息
                delete it_0->second.pre_integration; //删掉这一帧的预积分信息
                it_0->second.pre_integration = nullptr; //置空这一帧的预积分信息
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                { //7.把滑窗内最老一帧以前的帧的预积分信息全删掉
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }
                //8.删掉滑窗内最老帧以前的所有帧(不包括最老帧)，和最老帧
                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else // f (marginalization_flag == MARGIN_NEW) 删除的是滑窗第10帧。
    {  // (1)取出最新一帧的信息
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                // (2) 当前帧和前一帧之间的 IMU 预积分转换为当前帧和前二帧之间的 IMU 预积分
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            // (3) 用最新一帧的信息覆盖上一帧信息
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            // (4) 因为已经把第11帧的信息覆盖了第10帧，所以现在把第11帧清除
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear(); 

            // (5) 滑窗
            slideWindowNew(); //为什么这里可以不对前一帧进行边缘化而是直接丢弃，原因就是当前帧和前一帧很相似。
        }                     //因此当前帧与地图点之间的约束和前一帧与地图点之间的约束是接近的，直接丢弃并不会造成整个约束关系丢失信息
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{ //因为已经把第11帧的信息覆盖了第10帧，所以现在把第11帧清除
    sum_of_front++; //统计一共有多少次merge滑窗第10帧的情况
    f_manager.removeFront(frame_count); //唯一用法：当最新一帧(11)不是关键帧时，用于merge滑窗内最新帧(10)(仅在slideWindowNew()出现过)
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++; //统计一共有多少次merge滑窗第一帧的情况

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0]; //滑窗原先的最老帧(被merge掉)的旋转(c->w)
        R1 = Rs[0] * ric[0];   //滑窗原先第二老的帧(现在是最老帧)的旋转(c->w)
        P0 = back_P0 + back_R0 * tic[0];  //滑窗原先的最老帧(被merge掉)的平移(c->w)
        P1 = Ps[0] + Rs[0] * tic[0];     //滑窗原先第二老的帧(现在是最老帧)的平移(c->w)
        f_manager.removeBackShiftDepth(R0, P0, R1, P1); //把首次在原先最老帧出现的特征点转移到原先第二老帧的相机坐标里(仅在slideWindowOld()出现过)
    }
    else
        f_manager.removeBack(); //当最新一帧是关键帧时，用于merge滑窗内最老帧(仅在slideWindowOld()出现过)
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp; //pose_graph里要被重定位的关键帧的时间戳
    relo_frame_index = _frame_index; //回环帧在pose_graph里的id
    match_points.clear();
    match_points = _match_points;   //两帧共同的特征点在回环帧上的归一化坐标
    prev_relo_t = _relo_t;          //回环帧在自己世界坐标系的位姿(准)
    prev_relo_r = _relo_r;          //回环帧在自己世界坐标系的位姿(准)
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec()) //如果pose_graph里要被重定位的关键帧还在滑窗里
        {
            relo_frame_local_index = i;  //找到它的id
            relocalization_info = 1;     //告诉后端要重定位
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j]; //用这个关键帧的位姿(当前滑窗的世界坐标系下，不准)初始化回环帧的位姿
        }
    }
}


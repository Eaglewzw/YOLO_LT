#include "Kalman.hpp"




void Kalman::init(Eigen::Matrix<double, 6, 1> x_)
{
    X = x_;

    Q = Eigen::Matrix<double, 6, 6>::Identity() * 1;
    R = Eigen::Matrix<double, 6, 6>::Identity() * 0.3;


    P = Eigen::Matrix<double, 6, 6>::Identity() * 100;
    H = Eigen::Matrix<double, 6, 6>::Identity();

    K.setZero();
    F.setIdentity();
    F(0, 1) = 1;
    F(1, 2) = 1;

}



Eigen::Matrix<double, 6, 1>  Kalman::predict()
{
    x_predict = F * X;

    p_predict = F * P * F.transpose() + Q;

    return x_predict;
}



Eigen::Matrix<double, 6, 1>  Kalman::update(Eigen::Matrix<double, 6, 1> cur_pos)
{

    Eigen::Matrix<double, 6, 1> Z_t; 
    K = p_predict * H.transpose() * (H * p_predict * H.transpose() + R).inverse();


    X = x_predict + K * (cur_pos - H * x_predict);

    P = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * p_predict;
    
}



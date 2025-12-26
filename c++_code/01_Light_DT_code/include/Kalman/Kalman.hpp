#ifndef KALMAN_H
#define KALMAN_H


#include <Eigen/Eigen>

class Kalman
{
    private:
        /* data */

    public:
        Eigen::Matrix<double, 6, 1> X; //
        Eigen::Matrix<double, 6, 6> F; //
        Eigen::Matrix<double, 6, 6> Q; //
        Eigen::Matrix<double, 6, 6> R; //
        Eigen::Matrix<double, 6, 6> P; //
        Eigen::Matrix<double, 6, 6> H; //
        Eigen::Matrix<double, 6, 6> K; //


        Eigen::Matrix<double, 6, 6> p_predict;

        Eigen::Matrix<double, 6, 1> x_predict;



        Kalman(/* args */);

        void init(Eigen::Matrix<double, 6, 1> x_);

        Eigen::Matrix<double, 6, 1> predict();

        Eigen::Matrix<double, 6, 1> update(Eigen::Matrix<double, 6, 1> cur_pos);

        ~Kalman();
};


Kalman::Kalman(/* args */)
{
}

Kalman::~Kalman()
{
}


#endif

#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>

#include "p3p/p3p.h"
#include "hungarian.h"
#include "dbscan/dbscan.hpp"

const float n_thresh = 0.1;

struct wmr_distortion_6KT // from monado wmr_config.h
{
//    enum wmr_distortion_model model;

    /* The config provides 15 float values: */
    union {
        struct
        {
            float cx, cy;         /* Principal point */
            float fx, fy;         /* Focal length */
            float k[6];           /* Radial distortion coefficients */
            float dist_x, dist_y; /* Center of distortion */
            float p2, p1;         /* Tangential distortion parameters */
            float metric_radius;  /* Metric radius */
        } params;
        float v[15];
    };
};


cv::Ptr<cv::SimpleBlobDetector> blob_detector = nullptr;

auto get_blob_detector() {
    if (blob_detector == nullptr) {
        auto params = cv::SimpleBlobDetector::Params();
        params.blobColor = 255;
        params.minThreshold = 20;
        params.maxThreshold = 255;
        params.minDistBetweenBlobs = 1.0f;
        params.filterByArea = true;
        params.filterByCircularity = true;
        params.filterByConvexity = true;
        params.filterByInertia = true;
        params.minArea = 1;
        params.maxArea = 100;
        params.minCircularity = 0.5;
        params.minConvexity = 0.7;
        params.minInertiaRatio = 0.3;
        params.maxInertiaRatio = 1.1;
        params.maxCircularity = 1.1;
        params.maxConvexity = 1.1;
        blob_detector = cv::SimpleBlobDetector::create(params);
    }
    return blob_detector;
}

auto keypoints_to_points(std::vector<cv::KeyPoint> keypoints) {
    auto points = std::vector<cv::Point2f>();
    for (auto &keypoint: keypoints) {
        points.emplace_back(keypoint.pt);
    }
    return points;
}

auto get_blobs(cv::Mat img) {
    auto detector = get_blob_detector();
    auto blobs = std::vector<cv::KeyPoint>();
    detector->detect(img, blobs);
    return blobs;
}

auto draw_img_keypoints(const cv::Mat& img, std::vector<cv::KeyPoint> keypoints, cv::Scalar color = cv::Scalar(0, 0, 255)) {
    auto img_with_keypoints = cv::Mat();
    cv::drawKeypoints(img, keypoints, img_with_keypoints, color,
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return img_with_keypoints;
}

auto points_to_keypoints(std::vector<cv::Point2f> points, float size=3) {
    auto keypoints = std::vector<cv::KeyPoint>();
    for (auto &point: points) {
        keypoints.emplace_back(point, size);
    }
    return keypoints;
}

auto draw_img_with_points(const cv::Mat& img, std::vector<cv::Point2f> points, cv::Scalar color = cv::Scalar(0, 0, 255), float size=3) {
    return draw_img_keypoints(img, points_to_keypoints(points, size), color);
}

struct controller_points {
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;
};

auto get_wmr_pointccloud(std::string path) {
    auto file = std::ifstream(path);
    auto json = nlohmann::json();
    file >> json;
    auto points = std::vector<cv::Point3f>();
    auto normals = std::vector<cv::Point3f>();

    auto leds = json["CalibrationInformation"]["ControllerLeds"];

    for (auto &led: leds) {
        auto pos = led["Position"];
        std::cout << pos << std::endl;
        points.emplace_back(pos[0], pos[1], pos[2]);
        auto normal = led["Normal"];
        normals.emplace_back(normal[0], normal[1], normal[2]);
    }

    return controller_points{points, normals};
}

auto get_wmr_pointccloud() {
    return get_wmr_pointccloud("/home/samyr/wmr_config.json");
}

struct camera_config {
    cv::Mat camera_matrix;
    std::vector<float> distortion_coefficients;
    cv::Point2f principal_point;
    cv::Point2f focal_lengths;
};

std::array<camera_config, 4> camera_configs_from_file(const std::string& filepath){
    auto file = std::ifstream(filepath);
    auto json = nlohmann::json();
    file >> json;
    auto cameras = json["Cameras"];
    for (int i = 0; i < 4; i++){
        auto config = camera_config();
        auto camera = cameras[i];
        auto intrinsic = camera["Intrinsics"];
        auto params = std::vector<float>();
        for (auto& param : intrinsic["ModelParameters"]){
            params.push_back(param);
        }
        // params are k1, k2, p1, p2, k3, k4, k5, k6, codx, cody, rpmax

    }
}

std::vector<std::vector<cv::KeyPoint>> cluster_controllers_dbscan(std::vector<cv::KeyPoint> keypoints){
    std::vector<point2> points;
    for (auto& keypoint : keypoints){
        points.push_back(point2{keypoint.pt.x, keypoint.pt.y});
    }
    auto clusters = dbscan(points, 50, 3);
    auto ret = std::vector<std::vector<cv::KeyPoint>>();
    for (auto& cluster : clusters){
        auto cluster_keypoints = std::vector<cv::KeyPoint>();
        for (auto& idx : cluster){
            cluster_keypoints.push_back(keypoints[idx]);
        }
        ret.push_back(cluster_keypoints);
    }
    // Now sort by x
    std::sort(ret.begin(), ret.end(), [](const std::vector<cv::KeyPoint>& a, const std::vector<cv::KeyPoint>& b){
        return a[0].pt.x < b[0].pt.x;
    });
    return ret;
}

std::vector<std::vector<cv::KeyPoint>> cluster_controllers(std::vector<cv::KeyPoint> keypoints) {
    if (keypoints.size() < 5){
        return std::vector<std::vector<cv::KeyPoint>>{keypoints};
    }
    // Let's first try 2-means clustering
    auto bestlabels = std::vector<int>();
    auto centers = std::vector<cv::Point2f>();
    cv::kmeans(keypoints_to_points(keypoints), 2, bestlabels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               10, cv::KMEANS_PP_CENTERS, centers);
    // Let's find the minimum distance between two points from different clusters
    for (int i = 0; i < keypoints.size(); ++i){
        keypoints[i].class_id = bestlabels[i];
    }
    auto centers_dist = cv::norm(centers[0] - centers[1]);
    auto dists = std::vector<double>();
    for (int i = 0; i < keypoints.size(); ++i){
        dists.push_back(cv::norm(keypoints[i].pt - centers[bestlabels[i]]));
    }
    double min_dist_inter = 99999999;
    double avg_dist_intra = 0;
    double intra_cnt;
    for(auto& p1 : keypoints){
        for(auto& p2 : keypoints){
            if (p1.pt == p2.pt){
                continue;
            }
            if (p1.class_id == p2.class_id){
                avg_dist_intra += cv::norm(p1.pt - p2.pt);
                intra_cnt += 1;
                continue;
            }
            min_dist_inter = std::min(min_dist_inter, cv::norm(p1.pt - p2.pt));
        }
    }
    avg_dist_intra /= intra_cnt;

    if (min_dist_inter < 2 * avg_dist_intra){
        return std::vector<std::vector<cv::KeyPoint>>{keypoints};
    }

    auto ret = std::vector<std::vector<cv::KeyPoint>>(2);
    for (int i = 0; i < keypoints.size(); ++i){
        ret[bestlabels[i]].push_back(keypoints[i]);
    }

    if (centers[0].x > centers[1].x){
        std::swap(ret[0], ret[1]);
    }

    return ret;
}

auto get_camera_config() {
    std::vector<float> dist_coeffs = {-0.1769671145276103, 0.031858668906074486, -0.0001895582986086339,
                                      -0.0007531152703272859};
//    auto principal_point = cv::Point2f(250.29995368663705, 323.39904556782017);
    auto principal_point = cv::Point2f(323.39904556782017, 250.29995368663705);
    auto focal_lengths = cv::Point2f(267.03661387759524, 267.2191156905666);
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = focal_lengths.x;
    camera_matrix.at<double>(1, 1) = focal_lengths.y;
    camera_matrix.at<double>(0, 2) = principal_point.x;
    camera_matrix.at<double>(1, 2) = principal_point.y;
    return camera_config{camera_matrix, dist_coeffs, principal_point, focal_lengths};
}

auto get_camera_config_file(const std::string& filepath, int idx){
    std::ifstream file(filepath);
    auto json = nlohmann::json();
    file >> json;
    std::cout << json.dump() << std::endl;
    auto& cameras = json["CalibrationInformation"]["Cameras"];
    auto& camera = cameras[idx];
    auto& intrinsic = camera["Intrinsics"];
    std::cout << cameras.dump() << std::endl;
    std::cout << camera.dump() << std::endl;
    std::cout << intrinsic.dump() << std::endl;
    wmr_distortion_6KT dist;
    for (int i = 0; i < 15; i++){
        dist.v[i] = intrinsic["ModelParameters"][i];
    }
    int height, width;
    height = camera["SensorHeight"];
    width = camera["SensorWidth"];
    auto principal_point = cv::Point2f(dist.params.cx * width, dist.params.cy * height);
    auto focal_lengths = cv::Point2f(dist.params.fx * width, dist.params.fy * height);
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = focal_lengths.x;
    camera_matrix.at<double>(1, 1) = focal_lengths.y;
    camera_matrix.at<double>(0, 2) = principal_point.x;
    camera_matrix.at<double>(1, 2) = principal_point.y;
    auto dist_coeffs = std::vector<float>(8, 0);
    dist_coeffs[0] = dist.params.k[0];
    dist_coeffs[1] = dist.params.k[1];
    dist_coeffs[2] = dist.params.p1 * 2;
    dist_coeffs[3] = dist.params.p2 * 2;
    dist_coeffs[4] = dist.params.k[2];
    dist_coeffs[5] = dist.params.k[3];
    dist_coeffs[6] = dist.params.k[4];
    dist_coeffs[7] = dist.params.k[5];
    return camera_config{camera_matrix, dist_coeffs, principal_point, focal_lengths};
}

auto check_repeat(std::vector<int> idxs, int n) {
    auto counts = std::vector<int>(n, 0);
    for (auto idx: idxs) {
        counts[idx] += 1;
    }
    for (auto count: counts) {
        if (count > 1) {
            return true;
        }
    }
    return false;
}

auto get_k_permutation_idx(int k, int n) {
    std::vector<std::vector<int>> idxs;
    for (int i = 0; i < pow(n, k); ++i) {
        auto idx = std::vector<int>();
        auto tmp = i;
        for (int j = 0; j < k; ++j) {
            idx.push_back(tmp % n);
            tmp /= n;
        }
        if (!check_repeat(idx, n)) {
            idxs.push_back(idx);
        }
    }
    return idxs;
}

float fast_score(const std::vector<cvl::Vector3<float>>& a, const std::vector<cvl::Vector3<float>>& b){
    float dist_sum = 0;
    for (int i = 0; i < a.size(); ++i) {
        float min_dist = INT_MAX;
        for (int j = 0; j < b.size(); ++j) {
            float dist = (a[i] - b[j]).norm();
            if (dist < min_dist){
                min_dist = dist;
            }
        }
        dist_sum += min_dist;
    }
    return dist_sum / std::min(a.size(), b.size());
}

float fast_score(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b){
    float dist_sum = 0;
    for (int i = 0; i < a.size(); ++i) {
        float min_dist = INT_MAX;
        for (int j = 0; j < b.size(); ++j) {
            float dist = norm((a[i] - b[j]));
            if (dist < min_dist){
                min_dist = dist;
            }
        }
        dist_sum += min_dist;
    }
    return dist_sum / std::min(a.size(), b.size());
}

std::vector<int> assign(const std::vector<cvl::Vector3<float>>& a, const std::vector<cvl::Vector3<float>>& b){
    auto cost_matrix = std::vector<std::vector<int>>(a.size()+1, std::vector<int>(b.size()+1));
    for (int i = 0; i < a.size(); i++){
        for (int j = 0; j < b.size(); j++){
            cost_matrix[i+1][j+1] = (a[i] - b[j]).norm();
        }
    }
    auto solution = hungarian(cost_matrix);
    return solution.idxs;
}

float hungarian_score(const std::vector<cvl::Vector3<float>>& a, const std::vector<cvl::Vector3<float>>& b){
    auto cost_matrix = std::vector<std::vector<int>>(a.size(), std::vector<int>(b.size()));
    for (int i = 0; i < a.size(); i++){
        for (int j = 0; j < b.size(); j++){
            cost_matrix[i][j] = (a[i] - b[j]).norm();
        }
    }
    auto solution = hungarian(cost_matrix);
    return solution.cost;
}

std::vector<int> assign_closest(const std::vector<cvl::Vector3<float>>& a, const std::vector<cvl::Vector3<float>>& b){
    auto cost_matrix = std::vector<std::vector<double>>(a.size(), std::vector<double>(b.size()));
    for (int i = 0; i < a.size(); i++){
        for (int j = 0; j < b.size(); j++){
            cost_matrix[i][j] = (a[i] - b[j]).norm();
        }
    }
    std::vector<int> assignment(a.size(), -1);
    std::vector<int> taken(b.size(), 0);
    for (int i = 0; i < a.size(); i++){
        double min_dist = INT_MAX;
        int min_idx = -1;
        for (int j = 0; j < b.size(); j++){
            if (taken[j] == 0 && cost_matrix[i][j] < min_dist){
                min_dist = cost_matrix[i][j];
                min_idx = j;
            }
        }
        assignment[i] = min_idx;
        taken[min_idx] = 1;
    }
    return assignment;
}

std::vector<int> assign_some(const std::vector<cvl::Vector3<float>>& a, const std::vector<cvl::Vector3<float>>& b, int free){
    auto cost_matrix = std::vector<std::vector<int>>(a.size()+1, std::vector<int>(b.size()+1+free));
    for (int i = 0; i < a.size(); i++){
        for (int j = 0; j < b.size(); j++){
            cost_matrix[i+1][j+1] = (a[i] - b[j]).norm();
        }
        for (int j = 0; j < free; j++){
            cost_matrix[i+1][j+1+b.size()] = 0;
        }
    }
    auto solution = hungarian(cost_matrix);
    for (int i = 0; i < a.size(); i++){
        if (solution.idxs[i] >= b.size()){
            solution.idxs[i] = -1;
        }
    }
    return solution.idxs;
}

auto cvl_to_cv(const cvl::Vector3<float>& point){
    return cv::Point2f(point[0], point[1]);
}

auto to_homogenous_coords(const std::vector<cv::Point2f>& points, const camera_config& config){
    auto points_homogenous = std::vector<cv::Point3f>();
    for (auto& point: points){
        points_homogenous.emplace_back((point - config.principal_point).x / config.focal_lengths.x,
                                       (point - config.principal_point).y / config.focal_lengths.y, 1);
    }
    return points_homogenous;
}
auto backwards_normal_idx(const std::vector<cvl::Vector3<float>>& points, const std::vector<cvl::Vector3<float>>& normals, cvl::Matrix<float, 3, 3> R, cvl::Vector3f T, float tolerance = n_thresh){
    auto idxs = std::vector<int>();
    for (int i = 0; i < normals.size(); ++i) {
        auto normal = R * normals[i];
        auto point = R * points[i] + T;
        auto dot = normal.normalized().dot(point.normalized());
//        std::cout << i << " : " << dot << std::endl;
        if (dot < tolerance){
            idxs.push_back(i);
        }
    }
    return idxs;
}

auto refine(const std::vector<cvl::Vector3<float>>& imgpoints_homog, const std::vector<cvl::Vector3<float>>& worldpoints,
            std::array<int, 3> sampled_idx, cvl::Matrix<float, 3, 3> R, cvl::Vector3<float> T,
            const camera_config& config){
    // Let's calculate the distance between the reprojected points and the image points, and iteratively keep the least ambiguous
    std::vector<cvl::Vector3<float>> reprojected_points(worldpoints.size());
    for (const auto& point : worldpoints) {
        reprojected_points.push_back(R * point + T);
    }
    for (auto& point : reprojected_points) {
        point /= point[2];
    }
    for (auto& point : reprojected_points) {
        point[0] *= config.focal_lengths.x;
        point[1] *= config.focal_lengths.y;
        point[0] += config.principal_point.x;
        point[1] += config.principal_point.y;
    }


}


auto twister = std::mt19937{std::random_device{}()};

auto sqpnp_refine(const std::vector<cv::Point2f>& imgpoints, const std::vector<cv::Point3f>& worldpoints, std::vector<int> img2world, const camera_config& config) {
    cv::Vec3f rvec, tvec;
    std::vector<cv::Point2f> imgpoints_kept;
    std::vector<cv::Point3f> worldpoints_kept;
    for (int i = 0; i < img2world.size(); ++i) {
        if (img2world[i] != -1){
            imgpoints_kept.push_back(imgpoints[i]);
            worldpoints_kept.push_back(worldpoints[img2world[i]]);
        }
    }
    cv::solvePnP(worldpoints_kept, imgpoints_kept, config.camera_matrix, config.distortion_coefficients, rvec, tvec, false, cv::SOLVEPNP_SQPNP);
    return std::make_tuple(rvec, tvec, imgpoints_kept, worldpoints_kept);
}

auto reproject_score(std::vector<cv::Point2f> imgpoints, const std::vector<cv::Point3f>& worldpoints, const camera_config& config, const cv::Mat1f& R, const cv::Vec3f& tvec) {
    std::vector<cv::Point2f> reprojected_points;
    for(auto& point : worldpoints){
        cv::Mat1f rp_pt = (R * cv::Mat(point) + cv::Mat(tvec));
        cv::Vec3f reprojected_point = cv::Vec3f(rp_pt.reshape(3).at<cv::Vec3f>());
        reprojected_point /= reprojected_point[2];
        reprojected_point[0] *= config.focal_lengths.x;
        reprojected_point[1] *= config.focal_lengths.y;
        reprojected_point[0] += config.principal_point.x;
        reprojected_point[1] += config.principal_point.y;
        reprojected_points.emplace_back(reprojected_point[0], reprojected_point[1]);
    }
    return std::make_tuple(fast_score(imgpoints, reprojected_points), reprojected_points);
}

auto draw_assignment(const std::vector<cv::Point2f>& imgpoints, const std::vector<cv::Point2f>& reprojected_points,  const cv::Mat img){
    auto img_with_points = img.clone();

    for (int i = 0; i < imgpoints.size(); ++i) {
        cv::circle(img_with_points, reprojected_points[i], 3, cv::Scalar(255, 255, 0), 1);
        cv::line(img_with_points, imgpoints[i], reprojected_points[i], cv::Scalar(0, 0, 255));
    }
//    cv::imshow("img_with_points", img_with_points);
//    cv::waitKey(0);
    return img_with_points;
}


auto generate_assignation(const std::vector<cvl::Vector3f>& imgpoints, const std::vector<cvl::Vector3f>& worldpoints,
                          const std::vector<cvl::Vector3f>& normals, const camera_config& config,
                          const cvl::Matrix<float, 3, 3>& R, const cvl::Vector3<float>& T, cv::Mat img){
    // First, filter out worldpoints by normals
    auto idxs = backwards_normal_idx(worldpoints, normals, R, T);
    // Then, reproject these points
    auto reprojected_points = std::vector<cvl::Vector3f>();
    for (const auto& idx : idxs) {
        reprojected_points.push_back(R * worldpoints[idx] + T);
    }
    for (auto& point : reprojected_points) {
        point /= point[2];
    }
    for (auto& point : reprojected_points) {
        point[0] *= config.focal_lengths.x;
        point[1] *= config.focal_lengths.y;
        point[0] += config.principal_point.x;
        point[1] += config.principal_point.y;
        cv::circle(img, cv::Point2f(point[0], point[1]), 3, cv::Scalar(0, 128, 255), 1);
    }
    // Then, assign the reprojected points to the image points
    auto assignment = assign(imgpoints, reprojected_points);
    // Now, translate these to the original imgpoints idxs
    auto assignment_orig = std::vector<int>(imgpoints.size(), -1);
    for (int i = 1; i < assignment.size(); ++i) {
        assignment_orig[i-1] = idxs[assignment[i]-1];
    }

    auto score = fast_score(imgpoints, reprojected_points);

    std::cout << "Score: " << score << std::endl;
    return assignment_orig;
}

auto
bruteforce_match_pnp(std::vector<cv::Point2f> imgpoints, const std::vector<cv::Point3f>& worldpoints, const std::vector<cv::Point3f>& normals, const camera_config& config, const cv::Mat img) {
    // Choose 3 points from the image at random
    auto normals_cvl = std::vector<cvl::Vector3<float>>();
    for (auto &normal: normals) {
        normals_cvl.emplace_back(normal.x, normal.y, normal.z);
    }
    std::vector<cv::Point2f> imgpoints_sample;
    auto imgpoints_orig = imgpoints;
    std::vector<cvl::Vector3<float>> imgpoints_orig_cvl;
    for (auto& point : imgpoints){
        imgpoints_orig_cvl.emplace_back(point.x, point.y, 1);
    }
    for (auto& point : imgpoints){
        point = (point - config.principal_point);
        point.x /= config.focal_lengths.x;
        point.y /= config.focal_lengths.y;
    }
    std::sample(imgpoints.begin(), imgpoints.end(), std::back_inserter(imgpoints_sample), 3,
                twister);
    cvl::Vector3<float> imgpoints_sample_cvl[3] = {
            cvl::Vector3<float>(imgpoints_sample[0].x, imgpoints_sample[0].y, 1).normalized(),
            cvl::Vector3<float>(imgpoints_sample[1].x, imgpoints_sample[1].y, 1).normalized(),
            cvl::Vector3<float>(imgpoints_sample[2].x, imgpoints_sample[2].y, 1).normalized(),
    };
    auto imgpoints_cvl = std::vector<cvl::Vector3<float>>();
    for (auto &imgpoint: imgpoints) {
        imgpoints_cvl.emplace_back(imgpoint.x, imgpoint.y, 1);
    }
    // Then try to match them to all length 3 subsets of the world points
    auto idxs = get_k_permutation_idx(3, worldpoints.size());
    auto worldpoints_cvl = std::vector<cvl::Vector3<float>>();
    for (auto &worldpoint: worldpoints) {
        worldpoints_cvl.emplace_back(worldpoint.x, worldpoint.y, worldpoint.z);
    }


    auto rvecs = std::vector<cv::Mat>();
    auto tvecs = std::vector<cv::Mat>();
    int scount = 0;
    std::vector<float> scores(idxs.size(), INT_MAX);
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<cvl::Matrix<float, 3, 3>> best_Rs_all(idxs.size());
    std::vector<cvl::Vector3<float>> best_Ts_all(idxs.size());
# pragma omp parallel for schedule(dynamic, 1) num_threads(8)
    for (int i = 0; i < idxs.size(); i++) {
        auto Rs = cvl::Vector<cvl::Matrix<float, 3, 3>, 4>();
        auto Ts = cvl::Vector<cvl::Vector3<float>, 4>();
        const auto& idx = idxs[i];
        auto best_Rs = Rs[0];
        auto best_Ts = Ts[0];
        float best_score = INT_MAX;
        std::vector<cvl::Vector3<float>> reprojected_points(worldpoints_cvl.size());

        int sols = cvl::p3p(imgpoints_sample_cvl[0], imgpoints_sample_cvl[1], imgpoints_sample_cvl[2],
                               worldpoints_cvl[idx[0]], worldpoints_cvl[idx[1]], worldpoints_cvl[idx[2]],
                               Rs, Ts);

        double iter_scores[4] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};

        for (int j = 0; j < sols; ++j) {
            reprojected_points.clear();
            for (int k = 0; k < worldpoints_cvl.size(); ++k) {
                if ((Rs[j] * normals_cvl[k]).normalized().dot(Rs[j] * worldpoints_cvl[k] + Ts[j]) > n_thresh) {
                    continue;
                }
                reprojected_points.push_back(Rs[j] * worldpoints_cvl[k] + Ts[j]);
            }
            for (auto& point : reprojected_points) {
                point /= point[2];
            }
            for (auto& point : reprojected_points) {
                point[0] *= config.focal_lengths.x;
                point[1] *= config.focal_lengths.y;
                point[0] += config.principal_point.x;
                point[1] += config.principal_point.y;
            }


            iter_scores[j] = fast_score(imgpoints_orig_cvl, reprojected_points);
        }
        int best_idx = std::min_element(iter_scores, iter_scores + sols) - iter_scores;
        scores[i] = iter_scores[best_idx];
        best_Ts_all[i] = Ts[best_idx];
        best_Rs_all[i] = Rs[best_idx];
    }

    // Sort idxs by score
    std::vector<int> idxs_sorted(idxs.size());
    std::iota(idxs_sorted.begin(), idxs_sorted.end(), 0);
    std::sort(idxs_sorted.begin(), idxs_sorted.end(), [&scores](int i1, int i2) {return scores[i1] < scores[i2];});
    auto best_idx = idxs_sorted[0];
    auto best_Rs = best_Rs_all[best_idx];
    auto best_Ts = best_Ts_all[best_idx];
    auto best_Rs_cv = cv::Mat(3, 3, CV_32F, best_Rs.data());
    auto best_T_cv = cv::Mat(3, 1, CV_32F, best_Ts.data());
    auto best_R_cv = cv::Mat();
    cv::Rodrigues(best_Rs_cv, best_R_cv);
    float best_score = scores[best_idx];
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "--" << scount << " out of " << idxs.size() << " best_ " << best_score << " time " << duration.count() << "us" << std::endl;

    // Rescore, to debug

    {
        auto reprojected_points = std::vector<cvl::Vector3<float>>();
        for (int k = 0; k < worldpoints_cvl.size(); ++k) {
            if ((best_Rs * normals_cvl[k]).dot(best_Rs * worldpoints_cvl[k] + best_Ts) > n_thresh) {
                continue;
            }
            reprojected_points.push_back(best_Rs * worldpoints_cvl[k] + best_Ts);
        }
        for (auto& point : reprojected_points) {
            point /= point[2];
        }
        for (auto& point : reprojected_points) {
            point[0] *= config.focal_lengths.x;
            point[1] *= config.focal_lengths.y;
            point[0] += config.principal_point.x;
            point[1] += config.principal_point.y;
        }
        auto score = fast_score(imgpoints_orig_cvl, reprojected_points);
        std::cout << "Score: " << score << std::endl;
    }



    auto asignation = generate_assignation(imgpoints_orig_cvl, worldpoints_cvl, normals_cvl, config, best_Rs, best_Ts, img);
    auto refined = sqpnp_refine(imgpoints_orig, worldpoints, asignation, config);
    auto rvec_ = std::get<0>(refined);
    auto tvec_ = std::get<1>(refined);
    auto imgpoints_kept = std::get<2>(refined);
    auto worldpoints_kept = std::get<3>(refined);
    float score;
    cv::Mat1f rMat;
    cv::Rodrigues(rvec_, rMat);
    std::vector<cv::Point2f> reproj_points;
    std::tie(score, reproj_points) = reproject_score(imgpoints_orig, worldpoints_kept, config, rMat, cv::Mat1f(tvec_));
//    std::tie(score, reproj_points) = reproject_score(imgpoints_orig, worldpoints_kept, config, best_Rs_cv, best_T_cv);
    draw_assignment(imgpoints_kept, reproj_points, img);
    std::cout << "Score: " << score << std::endl;
    return score;
}


std::vector<std::string> list_all_files(const std::string& folder_path){
    auto files = std::vector<std::string>();
    for (const auto& object : std::filesystem::directory_iterator(folder_path)){
        files.push_back(object.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

void get_frame_capture(const std::string& path, std::array<cv::Mat, 4>& frames){
    auto file_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    // We have four cameras, so we need to split the image into four.
    // We also omit the first line
    auto tmp_size = cv::Size(640, 480);
    auto tmp_img = cv::Mat();
    auto offset = cv::Point(0, 1);
    for (int i = 0; i < 4; ++i) {
        auto roi = cv::Rect(offset, tmp_size);
        tmp_img = file_img(roi);
        frames[i] = tmp_img.clone();
        offset.x += 640;
    }

}

int main() {
    auto frame_path = "/home/samyr/Downloads/2023-04-07-windows-short-beatsaber-session/frames";
    auto files = list_all_files(frame_path);
    auto frames = std::array<cv::Mat, 4>();

    auto cloud_l = get_wmr_pointccloud("/home/samyr/Downloads/2023-04-07-windows-short-beatsaber-session/configs/reverb_g2_left.json");
    auto cloud_r = get_wmr_pointccloud("/home/samyr/Downloads/2023-04-07-windows-short-beatsaber-session/configs/reverb_g2_right.json");
    auto camera_config_og = get_camera_config_file("/home/samyr/Downloads/2023-04-07-windows-short-beatsaber-session/configs/reverbg2.json", 1);

    float score_sum = 0;
    int score_count = 0;

    for (auto& f : files){
        auto camera_config = camera_config_og;

        get_frame_capture(f, frames);
        frames[1] = frames[1] - 5;
        frames[1] = frames[1] * 6;


        // lets undistort
        cv::undistort(frames[1].clone(), frames[1], get_camera_config().camera_matrix, get_camera_config().distortion_coefficients);

        // Remove distortion from config now
        camera_config.distortion_coefficients = {0, 0, 0, 0};

        auto blobs = get_blobs(frames[1]);
        // Try to cluster blobs
        auto clusters = cluster_controllers_dbscan(blobs);
        std::cout << clusters.size() << std::endl;

        if (clusters.size() == 0)
            continue;


        auto img_with_keypoints = draw_img_keypoints(frames[1], clusters[0], cv::Scalar(0, 255, 0));
        if (clusters.size() > 1)
            img_with_keypoints = draw_img_keypoints(img_with_keypoints, clusters[1], cv::Scalar(255, 0, 0));

        auto img_track = img_with_keypoints.clone();

        for(int idx = 0; idx < clusters.size(); idx++){
            if (clusters[idx].size() <= 5)
                continue;
            // Try with the right controller first
            auto img_r = img_track.clone();
            auto imgpoints = keypoints_to_points(clusters[idx]);
            auto worldpoints = cloud_r.points;
            auto normals = cloud_r.normals;
            auto rvec = cv::Mat();
            auto tvec = cv::Mat();
            auto start = std::chrono::high_resolution_clock::now();
            auto score_r = bruteforce_match_pnp(imgpoints, worldpoints, normals, camera_config, img_r);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Total runtime " << duration.count() << "us" << std::endl;
            // Then the left controller
            auto img_l = img_with_keypoints.clone();
            worldpoints = cloud_l.points;
            normals = cloud_l.normals;
            auto score_l = bruteforce_match_pnp(imgpoints, worldpoints, normals, camera_config, img_l);
            std::cout << "Score: [R] " << score_r << " [L]" << score_l << std::endl;

            auto minscore = std::min(score_r, score_l);
            score_sum += minscore;
            score_count += 1;

            if (score_r < score_l){
                img_track = img_r;
            } else {
                img_track = img_l;
            }
        }

        cv::imshow("img", img_with_keypoints);
        cv::imshow("img_track", img_track);
        cv::waitKey(0);
    }
    std::cout << "Average reprojection error (average point distance): " << score_sum / score_count << std::endl;
}

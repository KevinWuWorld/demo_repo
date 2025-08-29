// synchronized_recorder_node_v2.cpp
// ROS1 (Noetic, catkin) + OpenCV.
// Records stereo/mono video (30 FPS) + high-rate kinematics (binary).
// Streams per arm: measured_js, measured_cp, measured_cv, setpoint_js, setpoint_cp.
// On shutdown: extract frames, timestamp from left start @ 30 FPS,
// and for each frame folder, write 5 nearest samples from all streams into kinematics.json,
// including image timestamps at the top.

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <jsoncpp/json/json.h>

#include <atomic>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

// ---------- CLI / config ----------
static std::string g_cam_base = "camera";
static bool g_use_left  = true;
static bool g_use_right = true;
static std::vector<std::string> g_arms; // from -a; if empty we'll default to all

static const int VIDEO_FPS = 30;
static const int FOURCC = cv::VideoWriter::fourcc('M','J','P','G'); // fast intra-only
static const size_t STREAMBUF_BYTES = 4*1024*1024; // buffered ofstream per kinematic file
static const std::string RUN_ROOT = "rec_v2";

static inline std::string topic_left()  { return "/"+g_cam_base+"/left/image_raw"; }
static inline std::string topic_right() { return "/"+g_cam_base+"/right/image_raw"; }

static inline std::string jsTopic(const std::string& arm){ return "/"+arm+"/measured_js"; }
static inline std::string cpTopic(const std::string& arm){ return "/"+arm+"/measured_cp"; }
static inline std::string cvTopic(const std::string& arm){ return "/"+arm+"/measured_cv"; }
static inline std::string jsSetTopic(const std::string& arm){ return "/"+arm+"/setpoint_js"; }
static inline std::string cpSetTopic(const std::string& arm){ return "/"+arm+"/setpoint_cp"; }

// ---------- arg parsing ----------
static void printUsage(){
  std::cout << "Usage: rosrun synchronized_recorder synchronized_recorder_node "
               "-c <camera base> -m <stereo|mono> [-d <left|right>] "
               "[-a PSM1] [-a PSM2] [-a PSM3] [-a ECM]\n";
}
static void parseArgs(int argc, char** argv){
  bool have_c=false, have_m=false;
  std::string mode, side;
  for (int i=1;i<argc;i++){
    std::string a = argv[i];
    if (a=="-c" && i+1<argc){ g_cam_base = argv[++i]; have_c=true; }
    else if (a=="-m" && i+1<argc){ mode = argv[++i]; have_m=true; }
    else if (a=="-d" && i+1<argc){ side = argv[++i]; }
    else if (a=="-a" && i+1<argc){ g_arms.emplace_back(argv[++i]); }
    else { printUsage(); exit(EXIT_FAILURE); }
  }
  if (!have_c || !have_m){ std::cerr<<"Missing -c or -m\n"; printUsage(); exit(EXIT_FAILURE); }
  if (mode=="stereo"){ g_use_left=true; g_use_right=true; if(!side.empty()){ std::cerr<<"-d not with stereo\n"; exit(EXIT_FAILURE);} }
  else if (mode=="mono"){
    if (side!="left" && side!="right"){ std::cerr<<"mono requires -d left|right\n"; exit(EXIT_FAILURE); }
    g_use_left  = (side=="left");
    g_use_right = (side=="right");
  } else { std::cerr<<"-m must be stereo|mono\n"; exit(EXIT_FAILURE); }

  // default arms if none given
  if (g_arms.empty()) g_arms = {"PSM1","PSM2","PSM3","ECM"};
}

// ---------- time helper ----------
struct T { int64_t sec{0}; int32_t nsec{0}; };
static inline double toSec(const T& t){ return double(t.sec) + double(t.nsec)*1e-9; }
static inline T fromDouble(double t){
  T out;
  out.sec  = static_cast<int64_t>(std::floor(t));
  double frac = t - static_cast<double>(out.sec);
  long long nn = static_cast<long long>(std::llround(frac * 1e9));
  if (nn >= 1000000000LL) { out.sec += 1; nn -= 1000000000LL; }
  if (nn < 0) { out.sec -= 1; nn += 1000000000LL; }
  out.nsec = static_cast<int32_t>(nn);
  return out;
}

// ---------- Binary logging format (little-endian) ----------
// JS record: [int64 sec][int32 nsec][uint32 npos][uint32 nvel][uint32 neff][npos doubles][nvel doubles][neff doubles]
// CP record: [int64 sec][int32 nsec][double px,py,pz,qx,qy,qz,qw]
// CV record: [int64 sec][int32 nsec][double vx,vy,vz, wx,wy,wz]
class FastBinWriter {
 public:
  explicit FastBinWriter(const std::string& path) {
    file_.open(path, std::ios::binary | std::ios::out);
    if(!file_) throw std::runtime_error("Failed to open " + path);
    buf_.resize(STREAMBUF_BYTES);
    file_.rdbuf()->pubsetbuf(buf_.data(), buf_.size());
  }
  ~FastBinWriter(){ file_.flush(); file_.close(); }
  FastBinWriter(const FastBinWriter&) = delete;
  FastBinWriter& operator=(const FastBinWriter&) = delete;

  void writeJS(const sensor_msgs::JointState& msg){
    const int64_t sec = msg.header.stamp.sec;
    const int32_t nsec = msg.header.stamp.nsec;
    const uint32_t np = static_cast<uint32_t>(msg.position.size());
    const uint32_t nv = static_cast<uint32_t>(msg.velocity.size());
    const uint32_t ne = static_cast<uint32_t>(msg.effort.size());
    file_.write(reinterpret_cast<const char*>(&sec),  sizeof(sec));
    file_.write(reinterpret_cast<const char*>(&nsec), sizeof(nsec));
    file_.write(reinterpret_cast<const char*>(&np),   sizeof(np));
    file_.write(reinterpret_cast<const char*>(&nv),   sizeof(nv));
    file_.write(reinterpret_cast<const char*>(&ne),   sizeof(ne));
    if(np) file_.write(reinterpret_cast<const char*>(msg.position.data()), np*sizeof(double));
    if(nv) file_.write(reinterpret_cast<const char*>(msg.velocity.data()), nv*sizeof(double));
    if(ne) file_.write(reinterpret_cast<const char*>(msg.effort.data()),   ne*sizeof(double));
  }
  void writeCP(const geometry_msgs::PoseStamped& msg){
    const int64_t sec = msg.header.stamp.sec;
    const int32_t nsec = msg.header.stamp.nsec;
    const double data[7] = {
      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
      msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
    };
    file_.write(reinterpret_cast<const char*>(&sec),  sizeof(sec));
    file_.write(reinterpret_cast<const char*>(&nsec), sizeof(nsec));
    file_.write(reinterpret_cast<const char*>(data), sizeof(data));
  }
  void writeCV(const geometry_msgs::TwistStamped& msg){
    const int64_t sec = msg.header.stamp.sec;
    const int32_t nsec = msg.header.stamp.nsec;
    const double data[6] = {
      msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
      msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
    };
    file_.write(reinterpret_cast<const char*>(&sec),  sizeof(sec));
    file_.write(reinterpret_cast<const char*>(&nsec), sizeof(nsec));
    file_.write(reinterpret_cast<const char*>(data), sizeof(data));
  }
 private:
  std::ofstream file_;
  std::vector<char> buf_;
};

// ---------------- video recorder ----------------
class VideoStreamRecorder {
 public:
  VideoStreamRecorder(const std::string& base_dir, const std::string& name)
    : base_dir_(base_dir), name_(name) {}
  bool push(const sensor_msgs::ImageConstPtr& msg){
    std::lock_guard<std::mutex> lk(m_);
    const double min_period = 1.0 / double(VIDEO_FPS);
    ros::Time stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    if (opened_) {
      if (stamp < last_written_) last_written_ = ros::Time(0);
      if ((stamp - last_written_).toSec() < min_period) return true; // drop to enforce 30 FPS
    }
    // convert to BGR
    cv_bridge::CvImageConstPtr cvp;
    try {
      if (msg->encoding == "bgr8") {
        cvp = cv_bridge::toCvShare(msg, "bgr8");
      } else if (msg->encoding == "rgb8") {
        auto rgb = cv_bridge::toCvShare(msg, "rgb8");
        cv::cvtColor(rgb->image, scratch_, cv::COLOR_RGB2BGR);
        cvp.reset(new cv_bridge::CvImage(rgb->header, "bgr8", scratch_));
      } else {
        cv_bridge::CvImagePtr cvcpy = cv_bridge::toCvCopy(msg, "bgr8");
        cvp = cvcpy;
      }
    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge error on %s: %s", name_.c_str(), e.what());
      return opened_;
    }
    if (!opened_) {
      start_time_ = stamp;
      path_ = base_dir_ + "/" + name_ + ".avi";
      writer_.open(path_, FOURCC, VIDEO_FPS, cvp->image.size(), true);
      if (!writer_.isOpened()) { ROS_ERROR("Failed to open VideoWriter: %s", path_.c_str()); return false; }
      opened_ = true;
      ROS_INFO("Opened %s (WxH=%dx%d) start=%d.%09d @ %d FPS",
               path_.c_str(), cvp->image.cols, cvp->image.rows,
               start_time_.sec, start_time_.nsec, VIDEO_FPS);
    }
    writer_.write(cvp->image);
    last_written_ = stamp;
    ++frames_written_;
    return true;
  }
  bool opened() const { return opened_; }
  ros::Time startTime() const { return start_time_; }
  ros::Time lastTime() const { return last_written_; }
  size_t frameCount() const { return frames_written_; }
  std::string path() const { return path_; }
  void release(){ std::lock_guard<std::mutex> lk(m_); if(opened_) writer_.release(); opened_=false; }
 private:
  std::string base_dir_, name_, path_;
  cv::VideoWriter writer_;
  bool opened_{false};
  ros::Time start_time_;
  ros::Time last_written_;
  size_t frames_written_{0};
  cv::Mat scratch_;
  std::mutex m_;
};

// --------- globals / state ----------
std::mutex g_mutex; // protects map creation
std::unique_ptr<VideoStreamRecorder> g_left_v, g_right_v;
std::string g_run_dir; // rec_v2/<sec>_<nsec>

struct StreamWriters {
  std::unique_ptr<FastBinWriter> js_meas;
  std::unique_ptr<FastBinWriter> cp_meas;
  std::unique_ptr<FastBinWriter> cv_meas;
  std::unique_ptr<FastBinWriter> js_set;
  std::unique_ptr<FastBinWriter> cp_set;
};
static std::map<std::string, StreamWriters> g_arm_writers;

static inline std::string makeRunDir(){
  ros::WallTime t = ros::WallTime::now();
  std::ostringstream oss; oss<<RUN_ROOT<<"/"<<t.sec<<"_"<<t.nsec;
  std::string dir = oss.str();
  std::error_code ec;
  std::filesystem::create_directories(dir+"/kin", ec);
  std::filesystem::create_directories(dir+"/meta", ec);
  std::filesystem::create_directories(dir+"/frames", ec);
  return dir;
}
static inline void writeStartMeta(const std::string& dir, const ros::Time& left_start, const ros::Time& right_start){
  Json::Value meta;
  meta["left_video_start"]["sec"]  = (Json::Value::Int64)left_start.sec;
  meta["left_video_start"]["nsec"] = (Json::Value::Int64)left_start.nsec;
  meta["right_video_start"]["sec"]  = (Json::Value::Int64)right_start.sec;
  meta["right_video_start"]["nsec"] = (Json::Value::Int64)right_start.nsec;
  std::ofstream f(dir+"/meta/start_times.json");
  Json::StreamWriterBuilder b; b["indentation"]="";
  std::unique_ptr<Json::StreamWriter> w(b.newStreamWriter());
  w->write(meta, &f);
}

// --------- video callbacks ----------
static void leftCb(const sensor_msgs::ImageConstPtr& msg){ if (g_left_v)  g_left_v->push(msg); }
static void rightCb(const sensor_msgs::ImageConstPtr& msg){ if (g_right_v) g_right_v->push(msg); }

// ensure writers for an arm exist
static void ensureArmWriters(const std::string& arm){
  std::lock_guard<std::mutex> lk(g_mutex);
  if (g_arm_writers.count(arm)) return;
  StreamWriters sw;
  sw.js_meas = std::make_unique<FastBinWriter>(g_run_dir + "/kin/" + arm + "_measured_js.bin");
  sw.cp_meas = std::make_unique<FastBinWriter>(g_run_dir + "/kin/" + arm + "_measured_cp.bin");
  sw.cv_meas = std::make_unique<FastBinWriter>(g_run_dir + "/kin/" + arm + "_measured_cv.bin");
  sw.js_set  = std::make_unique<FastBinWriter>(g_run_dir + "/kin/" + arm + "_setpoint_js.bin");
  sw.cp_set  = std::make_unique<FastBinWriter>(g_run_dir + "/kin/" + arm + "_setpoint_cp.bin");
  g_arm_writers.emplace(arm, std::move(sw));
}

// -------------- post-processing data structures --------------
struct JSRec { T t; std::vector<double> pos, vel, eff; };
struct CPRec { T t; double p[3]; double q[4]; };
struct CVRec { T t; double v[3]; double w[3]; };

static std::vector<JSRec> loadJS(const std::string& path){
  std::vector<JSRec> out; std::ifstream f(path, std::ios::binary); if(!f) return out;
  while(true){
    int64_t sec; int32_t nsec; uint32_t np,nv,ne;
    if(!f.read(reinterpret_cast<char*>(&sec),sizeof(sec))) break;
    if(!f.read(reinterpret_cast<char*>(&nsec),sizeof(nsec))) break;
    if(!f.read(reinterpret_cast<char*>(&np),sizeof(np))) break;
    if(!f.read(reinterpret_cast<char*>(&nv),sizeof(nv))) break;
    if(!f.read(reinterpret_cast<char*>(&ne),sizeof(ne))) break;
    JSRec r; r.t.sec=sec; r.t.nsec=nsec;
    r.pos.resize(np); r.vel.resize(nv); r.eff.resize(ne);
    if(np) f.read(reinterpret_cast<char*>(r.pos.data()), np*sizeof(double));
    if(nv) f.read(reinterpret_cast<char*>(r.vel.data()), nv*sizeof(double));
    if(ne) f.read(reinterpret_cast<char*>(r.eff.data()), ne*sizeof(double));
    out.push_back(std::move(r));
  } return out;
}
static std::vector<CPRec> loadCP(const std::string& path){
  std::vector<CPRec> out; std::ifstream f(path, std::ios::binary); if(!f) return out;
  while(true){
    int64_t sec; int32_t nsec; double data[7];
    if(!f.read(reinterpret_cast<char*>(&sec),sizeof(sec))) break;
    if(!f.read(reinterpret_cast<char*>(&nsec),sizeof(nsec))) break;
    if(!f.read(reinterpret_cast<char*>(data),sizeof(data))) break;
    CPRec r; r.t.sec=sec; r.t.nsec=nsec;
    r.p[0]=data[0]; r.p[1]=data[1]; r.p[2]=data[2];
    r.q[0]=data[3]; r.q[1]=data[4]; r.q[2]=data[5]; r.q[3]=data[6];
    out.push_back(r);
  } return out;
}
static std::vector<CVRec> loadCV(const std::string& path){
  std::vector<CVRec> out; std::ifstream f(path, std::ios::binary); if(!f) return out;
  while(true){
    int64_t sec; int32_t nsec; double data[6];
    if(!f.read(reinterpret_cast<char*>(&sec),sizeof(sec))) break;
    if(!f.read(reinterpret_cast<char*>(&nsec),sizeof(nsec))) break;
    if(!f.read(reinterpret_cast<char*>(data),sizeof(data))) break;
    CVRec r; r.t.sec=sec; r.t.nsec=nsec;
    r.v[0]=data[0]; r.v[1]=data[1]; r.v[2]=data[2];
    r.w[0]=data[3]; r.w[1]=data[4]; r.w[2]=data[5];
    out.push_back(r);
  } return out;
}

template <typename REC, typename GETTIME>
static std::vector<size_t> nearestK(const std::vector<REC>& v, double target, size_t K, GETTIME getTime){
  std::vector<size_t> idx; if(v.empty()||K==0) return idx;
  auto comp = [&](const REC& a, double t){ return getTime(a) < t; };
  auto it = std::lower_bound(v.begin(), v.end(), target, comp);
  size_t r = (size_t)std::distance(v.begin(), it);
  long L = (long)r-1, R = (long)r;
  while(idx.size() < K && (L>=0 || (size_t)R < v.size())){
    double dl = (L>=0) ? std::abs(getTime(v[(size_t)L]) - target) : std::numeric_limits<double>::infinity();
    double dr = ((size_t)R<v.size()) ? std::abs(getTime(v[(size_t)R]) - target) : std::numeric_limits<double>::infinity();
    if (dl <= dr){ idx.push_back((size_t)L); --L; } else { idx.push_back((size_t)R); ++R; }
  }
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
    return std::abs(getTime(v[a])-target) < std::abs(getTime(v[b])-target);
  });
  return idx;
}

// ------------- write JSON (now includes image stamps at the top) -------------
static void writeKinematicsJSON(const std::string& folder,
                                const std::map<std::string,std::vector<JSRec>>& js_meas,
                                const std::map<std::string,std::vector<CPRec>>& cp_meas,
                                const std::map<std::string,std::vector<CVRec>>& cv_meas,
                                const std::map<std::string,std::vector<JSRec>>& js_set,
                                const std::map<std::string,std::vector<CPRec>>& cp_set,
                                double t_frame,
                                const std::vector<std::string>& arms,
                                const T& img_left_stamp,
                                const std::optional<T>& img_right_stamp)
{
  Json::Value root;

  // Put image timestamps first (jsoncpp keeps insertion order in output).
  root["image_stamp_left"]["sec"]  = (Json::Value::Int64)img_left_stamp.sec;
  root["image_stamp_left"]["nsec"] = (Json::Value::Int64)img_left_stamp.nsec;
  if (img_right_stamp.has_value()){
    root["image_stamp_right"]["sec"]  = (Json::Value::Int64)img_right_stamp->sec;
    root["image_stamp_right"]["nsec"] = (Json::Value::Int64)img_right_stamp->nsec;
  }

  auto jsTime = [](const JSRec& r){ return toSec(r.t); };
  auto cpTime = [](const CPRec& r){ return toSec(r.t); };
  auto cvTime = [](const CVRec& r){ return toSec(r.t); };

  for (const auto& A : arms){
    Json::Value armBlk;

    auto dumpJSList = [&](const std::vector<JSRec>& vec){
      Json::Value arr(Json::arrayValue);
      auto idx = nearestK(vec, t_frame, 5, jsTime);
      for (auto i : idx){
        const auto& r = vec[i];
        Json::Value rec;
        rec["sec"]=(Json::Value::Int64)r.t.sec; rec["nsec"]=(Json::Value::Int64)r.t.nsec;
        Json::Value p(Json::arrayValue); for(double x: r.pos) p.append(x);
        Json::Value v(Json::arrayValue); for(double x: r.vel) v.append(x);
        Json::Value e(Json::arrayValue); for(double x: r.eff) e.append(x);
        rec["position"]=p; rec["velocity"]=v; rec["effort"]=e;
        arr.append(rec);
      }
      return arr;
    };
    auto dumpCPList = [&](const std::vector<CPRec>& vec){
      Json::Value arr(Json::arrayValue);
      auto idx = nearestK(vec, t_frame, 5, cpTime);
      for (auto i : idx){
        const auto& r = vec[i];
        Json::Value rec;
        rec["sec"]=(Json::Value::Int64)r.t.sec; rec["nsec"]=(Json::Value::Int64)r.t.nsec;
        Json::Value pos(Json::arrayValue); pos.append(r.p[0]); pos.append(r.p[1]); pos.append(r.p[2]);
        Json::Value ori(Json::arrayValue); ori.append(r.q[0]); ori.append(r.q[1]); ori.append(r.q[2]); ori.append(r.q[3]);
        rec["position"]=pos; rec["orientation"]=ori;
        arr.append(rec);
      }
      return arr;
    };
    auto dumpCVList = [&](const std::vector<CVRec>& vec){
      Json::Value arr(Json::arrayValue);
      auto idx = nearestK(vec, t_frame, 5, cvTime);
      for (auto i : idx){
        const auto& r = vec[i];
        Json::Value rec;
        rec["sec"]=(Json::Value::Int64)r.t.sec; rec["nsec"]=(Json::Value::Int64)r.t.nsec;
        Json::Value lin(Json::arrayValue); lin.append(r.v[0]); lin.append(r.v[1]); lin.append(r.v[2]);
        Json::Value ang(Json::arrayValue); ang.append(r.w[0]); ang.append(r.w[1]); ang.append(r.w[2]);
        rec["linear"]=lin; rec["angular"]=ang;
        arr.append(rec);
      }
      return arr;
    };

    if (js_meas.count(A) && !js_meas.at(A).empty()) armBlk["measured_js"] = dumpJSList(js_meas.at(A));
    if (cp_meas.count(A) && !cp_meas.at(A).empty()) armBlk["measured_cp"] = dumpCPList(cp_meas.at(A));
    if (cv_meas.count(A) && !cv_meas.at(A).empty()) armBlk["measured_cv"] = dumpCVList(cv_meas.at(A));
    if (js_set.count(A)  && !js_set.at(A).empty())  armBlk["setpoint_js"] = dumpJSList(js_set.at(A));
    if (cp_set.count(A)  && !cp_set.at(A).empty())  armBlk["setpoint_cp"] = dumpCPList(cp_set.at(A));

    root[A] = armBlk;
  }

  Json::StreamWriterBuilder b; b["indentation"]="";
  std::ofstream f(folder + "/kinematics.json");
  std::unique_ptr<Json::StreamWriter> w(b.newStreamWriter());
  w->write(root, &f);
}

// ---------------- post-processing ----------------
static void postProcess(const std::string& run_dir,
                        const std::string& left_path, const ros::Time& left_start,
                        const std::string& right_path,
                        const std::vector<std::string>& arms){
  ROS_INFO("Post-processing started...");

  // Load all streams per selected arm
  std::map<std::string,std::vector<JSRec>> all_js_meas, all_js_set;
  std::map<std::string,std::vector<CPRec>> all_cp_meas, all_cp_set;
  std::map<std::string,std::vector<CVRec>> all_cv_meas;

  for (const auto& A : arms){
    all_js_meas[A] = loadJS(run_dir+"/kin/"+A+"_measured_js.bin");
    all_cp_meas[A] = loadCP(run_dir+"/kin/"+A+"_measured_cp.bin");
    all_cv_meas[A] = loadCV(run_dir+"/kin/"+A+"_measured_cv.bin");
    all_js_set[A]  = loadJS(run_dir+"/kin/"+A+"_setpoint_js.bin");
    all_cp_set[A]  = loadCP(run_dir+"/kin/"+A+"_setpoint_cp.bin");
    ROS_INFO("%s: JS(meas)%zu CP(meas)%zu CV(meas)%zu  JS(set)%zu CP(set)%zu",
             A.c_str(),
             all_js_meas[A].size(), all_cp_meas[A].size(), all_cv_meas[A].size(),
             all_js_set[A].size(),  all_cp_set[A].size());
  }

  if (!std::filesystem::exists(left_path)) { ROS_WARN("Left video not found: %s", left_path.c_str()); return; }
  cv::VideoCapture capL(left_path);
  cv::VideoCapture capR(right_path);
  if(!capL.isOpened()){ ROS_WARN("Left video can't be opened: %s", left_path.c_str()); return; }

  const int totalL = (int)capL.get(cv::CAP_PROP_FRAME_COUNT);
  ROS_INFO("Left frames: %d", totalL);

  const double t0 = double(left_start.sec) + double(left_start.nsec)*1e-9;

  for (int i=0;i<totalL;i++){
    cv::Mat fL, fR;
    if(!capL.read(fL)) break;
    bool haveRight = false;
    if(capR.isOpened()) haveRight = capR.read(fR);

    // Per your rule, right timestamp == left timestamp
    const double t_frame = t0 + double(i) / double(VIDEO_FPS);
    const T leftStamp  = fromDouble(t_frame);
    const std::optional<T> rightStamp = haveRight ? std::optional<T>(leftStamp) : std::optional<T>{};

    // folder
    std::ostringstream oss; oss<<run_dir<<"/frames/frame_"<<i;
    std::string outdir = oss.str();
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);

    // images
    cv::imwrite(outdir + "/image_left.png",  fL);
    if (haveRight && !fR.empty()) cv::imwrite(outdir + "/image_right.png", fR);

    // kinematics (top includes image stamps)
    writeKinematicsJSON(outdir,
                        all_js_meas, all_cp_meas, all_cv_meas,
                        all_js_set,  all_cp_set,
                        t_frame, arms,
                        leftStamp, rightStamp);
  }

  ROS_INFO("Post-processing done: %s/frames", run_dir.c_str());
}

// ---------------- main ----------------
int main(int argc, char** argv){
  parseArgs(argc, argv);

  ros::init(argc, argv, "synchronized_recorder_v2");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  // run dir and video recorders
  g_run_dir = makeRunDir();
  if (g_use_left)  g_left_v  = std::make_unique<VideoStreamRecorder>(g_run_dir, "left");
  if (g_use_right) g_right_v = std::make_unique<VideoStreamRecorder>(g_run_dir, "right");

  // video subscriptions
  std::vector<image_transport::Subscriber> imgSubs;
  if (g_use_left) {
    ROS_INFO("Subscribing LEFT  : %s", topic_left().c_str());
    imgSubs.emplace_back(it.subscribe(topic_left(),  10, leftCb));
  }
  if (g_use_right) {
    ROS_INFO("Subscribing RIGHT : %s", topic_right().c_str());
    imgSubs.emplace_back(it.subscribe(topic_right(), 10, rightCb));
  }

  // kinematics subs (lambdas + hints)
  std::vector<ros::Subscriber> subs;
  for (const auto& arm : g_arms){
    // measured_js
    {
      auto cb = [arm](const sensor_msgs::JointState::ConstPtr& msg){
        ensureArmWriters(arm);
        g_arm_writers[arm].js_meas->writeJS(*msg);
      };
      subs.push_back(nh.subscribe<sensor_msgs::JointState>(jsTopic(arm), 1000, cb,
                      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay()));
      ROS_INFO("Sub JS (meas): %s", jsTopic(arm).c_str());
    }
    // measured_cp
    {
      auto cb = [arm](const geometry_msgs::PoseStamped::ConstPtr& msg){
        ensureArmWriters(arm);
        g_arm_writers[arm].cp_meas->writeCP(*msg);
      };
      subs.push_back(nh.subscribe<geometry_msgs::PoseStamped>(cpTopic(arm), 1000, cb,
                      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay()));
      ROS_INFO("Sub CP (meas): %s", cpTopic(arm).c_str());
    }
    // measured_cv (TwistStamped)
    {
      auto cb = [arm](const geometry_msgs::TwistStamped::ConstPtr& msg){
        ensureArmWriters(arm);
        g_arm_writers[arm].cv_meas->writeCV(*msg);
      };
      subs.push_back(nh.subscribe<geometry_msgs::TwistStamped>(cvTopic(arm), 1000, cb,
                      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay()));
      ROS_INFO("Sub CV (meas): %s", cvTopic(arm).c_str());
    }
    // setpoint_js
    {
      auto cb = [arm](const sensor_msgs::JointState::ConstPtr& msg){
        ensureArmWriters(arm);
        g_arm_writers[arm].js_set->writeJS(*msg);
      };
      subs.push_back(nh.subscribe<sensor_msgs::JointState>(jsSetTopic(arm), 1000, cb,
                      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay()));
      ROS_INFO("Sub JS (set) : %s", jsSetTopic(arm).c_str());
    }
    // setpoint_cp
    {
      auto cb = [arm](const geometry_msgs::PoseStamped::ConstPtr& msg){
        ensureArmWriters(arm);
        g_arm_writers[arm].cp_set->writeCP(*msg);
      };
      subs.push_back(nh.subscribe<geometry_msgs::PoseStamped>(cpSetTopic(arm), 1000, cb,
                      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay()));
      ROS_INFO("Sub CP (set) : %s", cpSetTopic(arm).c_str());
    }
  }

  ros::AsyncSpinner spinner(std::max(4u, std::thread::hardware_concurrency()));
  spinner.start();
  ROS_INFO("Recording... (Ctrl+C to stop)");
  ROS_INFO("\033[1;32m********** Be patient with the post-processing! **********\033[0m");
  ROS_INFO("\033[1;32m********** Post-processing takes around 3-5x the duration that was recorded! **********\033[0m");

  ros::waitForShutdown();

  // close video writers & write meta
  ros::Time left_start, right_start, left_last, right_last;
  size_t left_frames = 0, right_frames = 0;
  if (g_use_left && g_left_v){
    if (g_left_v->opened())  left_start  = g_left_v->startTime();
    left_last   = g_left_v->lastTime();
    left_frames = g_left_v->frameCount();
    g_left_v->release();
  }
  if (g_use_right && g_right_v){
    if (g_right_v->opened()) right_start = g_right_v->startTime();
    right_last   = g_right_v->lastTime();
    right_frames = g_right_v->frameCount();
    g_right_v->release();
  }
  if (g_use_left || g_use_right) writeStartMeta(g_run_dir, left_start, right_start);

  // print video frame counts + measured FPS
  if (g_use_left && left_frames > 0 && left_last > left_start){
    double dur = (left_last - left_start).toSec();
    double fps = (dur > 0.0) ? (double)left_frames / dur : 0.0;
    ROS_INFO("Left video : frames=%zu duration=%.6fs measured_fps=%.3f (target %d)",
             left_frames, dur, fps, VIDEO_FPS);
  } else if (g_use_left) {
    ROS_WARN("Left video : no frames recorded.");
  }
  if (g_use_right && right_frames > 0 && right_last > right_start){
    double dur = (right_last - right_start).toSec();
    double fps = (dur > 0.0) ? (double)right_frames / dur : 0.0;
    ROS_INFO("Right video: frames=%zu duration=%.6fs measured_fps=%.3f (target %d)",
             right_frames, dur, fps, VIDEO_FPS);
  } else if (g_use_right) {
    ROS_WARN("Right video: no frames recorded.");
  }

  // post-process
  const std::string left_path  = g_run_dir + "/left.avi";
  const std::string right_path = g_run_dir + "/right.avi";
  if (g_use_left && std::filesystem::exists(left_path)){
    if (left_start.isZero()) left_start = ros::Time::now();
    postProcess(g_run_dir, left_path, left_start, right_path, g_arms);
  } else {
    ROS_WARN("Skipping post-processing (no left video).");
  }

  ROS_INFO("All done.");
  return 0;
}

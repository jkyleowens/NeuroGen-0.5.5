#include "NeuroGen/VisualInterface.h"
#include "NeuroGen/NetworkConfig.h"
#include <iostream>
#include <chrono>

VisualInterface::VisualInterface(int width, int height)
    : target_width_(width),
      target_height_(height),
      detection_threshold_(0.5f),
      enable_preprocessing_(false),
      capture_active_(false) {}

VisualInterface::~VisualInterface() {
    stop_capture();
}

bool VisualInterface::initialize_capture() {
    if (!real_screen_capture_) {
        real_screen_capture_ = std::make_unique<RealScreenCapture>();
        if (!real_screen_capture_->initialize(target_width_, target_height_)) {
            std::cerr << "VisualInterface: failed to init screen capture" << std::endl;
            return false;
        }
    }
    if (!gui_detector_) gui_detector_ = std::make_unique<GUIElementDetector>();
    if (gui_detector_) gui_detector_->initialize();

    if (!ocr_processor_) {
        ocr_processor_ = std::make_unique<OCRProcessor>();
        ocr_processor_->initialize();
    }
    if (!visual_processor_) {
        NetworkConfig cfg;
        visual_processor_ = std::make_unique<BioVisualProcessor>("visual_processor", cfg, 64);
        visual_processor_->initialize();
    }
    return true;
}

void VisualInterface::start_continuous_capture() {
    if (capture_active_) return;
    capture_active_ = true;
    capture_thread_ = std::thread(&VisualInterface::capture_loop, this);
}

void VisualInterface::stop_capture() {
    capture_active_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
}

std::vector<float> VisualInterface::capture_and_process_screen() {
    if (!real_screen_capture_) return {};
    cv::Mat img = real_screen_capture_->captureScreen();
    if (img.empty()) return {};
    {
        std::lock_guard<std::mutex> lock(screen_mutex_);
        current_screen_ = img.clone();
    }
    if (visual_processor_) {
        return visual_processor_->processPixels(img);
    }
    return {};
}

std::vector<ScreenElement> VisualInterface::detect_screen_elements() {
    if (!gui_detector_ || !real_screen_capture_) return {};
    cv::Mat img = real_screen_capture_->captureScreen();
    return gui_detector_->detectElements(img);
}

void VisualInterface::update_element_detection() {
    auto elems = detect_screen_elements();
    std::lock_guard<std::mutex> lock(screen_mutex_);
    detected_elements_ = elems;
}

ScreenElement VisualInterface::find_element_by_type(const std::string& type) const {
    std::lock_guard<std::mutex> lock(screen_mutex_);
    for (const auto& e : detected_elements_) {
        if (e.type == type) return e;
    }
    return {};
}

bool VisualInterface::is_element_visible(const ScreenElement& element) const {
    return element.confidence >= detection_threshold_;
}

std::vector<float> VisualInterface::get_visual_features(const ScreenElement& element) const {
    cv::Mat frame;
    {
        std::lock_guard<std::mutex> lock(screen_mutex_);
        if (current_screen_.empty()) return {};
        frame = current_screen_.clone();
    }
    cv::Rect r(element.x, element.y, element.width, element.height);
    r &= cv::Rect(0,0,frame.cols, frame.rows);
    cv::Mat roi = frame(r);
    if (visual_processor_) {
        return visual_processor_->processPixels(roi);
    }
    return {};
}

cv::Mat VisualInterface::get_last_frame() const {
    std::lock_guard<std::mutex> lock(screen_mutex_);
    return current_screen_.clone();
}

cv::Mat VisualInterface::get_attention_map() const {
    // Placeholder implementation
    return cv::Mat();
}

std::vector<float> VisualInterface::extract_visual_features() const {
    std::lock_guard<std::mutex> lock(screen_mutex_);
    if (current_screen_.empty() || !visual_processor_) return {};
    return visual_processor_->processPixels(current_screen_);
}

void VisualInterface::apply_visual_feature_enhancement(std::vector<float>& features) const {
    float max_v = 0.f;
    for (float v : features) max_v = std::max(max_v, v);
    if (max_v > 0.f) {
        for (float& v : features) v /= max_v;
    }
}

void VisualInterface::send_to_visual_cortex(const cv::Mat& image) {
    std::lock_guard<std::mutex> lock(screen_mutex_);
    current_screen_ = image.clone();
    if (visual_processor_) {
        visual_features_ = visual_processor_->processPixels(current_screen_);
    }
}

void VisualInterface::capture_loop() {
    while (capture_active_) {
        capture_and_process_screen();
        update_element_detection();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void VisualInterface::preprocess_image() {}
void VisualInterface::extract_text_elements() {}
void VisualInterface::detect_interactive_elements() {}


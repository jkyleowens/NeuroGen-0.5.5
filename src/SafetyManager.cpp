#include "NeuroGen/SafetyManager.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include <iostream>

SafetyManager& SafetyManager::getInstance() {
    static SafetyManager instance;
    return instance;
}

void SafetyManager::enableGlobalSafety(bool enable) {
    global_safety_enabled_ = enable;
}

void SafetyManager::setScreenBounds(int width, int height) {
    screen_bounds_ = cv::Rect(0,0,width,height);
}

void SafetyManager::addForbiddenRegion(int x, int y, int width, int height) {
    forbidden_regions_.emplace_back(x,y,width,height);
}

void SafetyManager::setMaxActionsPerSecond(int max_actions) {
    max_actions_per_second_ = max_actions;
}

bool SafetyManager::checkRateLimit() const {
    auto now = std::chrono::steady_clock::now();
    while(!recent_actions_.empty() && std::chrono::duration_cast<std::chrono::seconds>(now - recent_actions_.front()).count() > 1) {
        recent_actions_.pop();
    }
    return static_cast<int>(recent_actions_.size()) < max_actions_per_second_;
}

bool SafetyManager::checkSpatialBounds(int x, int y) const {
    if(x < screen_bounds_.x || y < screen_bounds_.y || x > screen_bounds_.x + screen_bounds_.width || y > screen_bounds_.y + screen_bounds_.height)
        return false;
    for(const auto& rect : forbidden_regions_) {
        if(rect.contains(cv::Point(x,y))) return false;
    }
    return true;
}

bool SafetyManager::isActionSafe(const BrowsingAction& action) const {
    if(!global_safety_enabled_) return true;
    if(action.type == ActionType::CLICK || action.type == ActionType::SCROLL) {
        if(!checkSpatialBounds(action.x_coordinate, action.y_coordinate)) return false;
    }
    return checkRateLimit();
}

void SafetyManager::recordAction(const BrowsingAction& /*action*/) {
    recent_actions_.push(std::chrono::steady_clock::now());
}

#ifndef SAFETY_MANAGER_H
#define SAFETY_MANAGER_H
#include <functional>
#include <opencv2/core.hpp>
#include "NeuroGen/InputController.h"
#include <vector>
#include <chrono>
#include <queue>
#ifdef COUNT
#undef COUNT
#endif
#ifdef Status
#undef Status
#endif

struct BrowsingAction;

class SafetyManager {
public:
    static SafetyManager& getInstance();
    void enableGlobalSafety(bool enable);
    void setScreenBounds(int width, int height);
    void addForbiddenRegion(int x, int y, int width, int height);
    void setMaxActionsPerSecond(int max_actions);
    bool isActionSafe(const BrowsingAction& action) const;
    void recordAction(const BrowsingAction& action);
private:
    SafetyManager() = default;
    bool global_safety_enabled_ = false;
    std::vector<cv::Rect> forbidden_regions_;
    cv::Rect screen_bounds_;
    mutable std::queue<std::chrono::steady_clock::time_point> recent_actions_;
    int max_actions_per_second_ = 10;
    bool checkRateLimit() const;
    bool checkSpatialBounds(int x, int y) const;
};

#endif // SAFETY_MANAGER_H

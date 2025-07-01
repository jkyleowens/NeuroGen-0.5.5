// ============================================================================
// SCREEN ELEMENT DEFINITION HEADER
// File: include/NeuroGen/ScreenElement.h
// ============================================================================

#ifndef SCREEN_ELEMENT_H
#define SCREEN_ELEMENT_H

#include <string>

/**
 * @brief Screen element structure for UI automation
 * 
 * This structure represents interactive elements detected on a screen
 * for automated interaction by the autonomous learning agent.
 */
struct ScreenElement {
    int id;                 // Unique identifier for the element
    std::string type;       // Type: "button", "textbox", "link", "image", etc.
    int x, y;              // Position coordinates (top-left corner)
    int width, height;     // Dimensions
    std::string text;      // Text content (if any)
    bool is_clickable;     // Whether the element can be clicked
    float confidence = 0.0f; // Detection confidence (0.0 - 1.0)
    
    // Default constructor
    ScreenElement() : id(0), x(0), y(0), width(0), height(0), is_clickable(false), confidence(0.0f) {}
    
    // Main constructor with optional confidence parameter
    ScreenElement(int _id, const std::string& _type, int _x, int _y, int _w, int _h, 
                  const std::string& _text, bool _clickable, float _conf = 0.8f)
        : id(_id), type(_type), x(_x), y(_y), width(_w), height(_h), 
          text(_text), is_clickable(_clickable), confidence(_conf) {}
};

#endif // SCREEN_ELEMENT_H

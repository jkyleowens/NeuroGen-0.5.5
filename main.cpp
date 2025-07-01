#include <NeuroGen/CentralController.h>
#include <NeuroGen/ScreenElement.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "--- ANIMA-Based Task Automation Simulation ---" << std::endl;
    std::cout << "---           System Bootstrapping         ---" << std::endl;

    // Create and initialize the central controller, which builds the entire modular network.
    CentralController controller;
    controller.initialize();

    // --- Simulation Run 1: Login Screen ---
    std::cout << "\n--- Starting Run 1: Encountering a Login Screen ---" << std::endl;

    // Simulate an initial screen state with a login button and a textbox.
    std::vector<ScreenElement> screen1;
    screen1.push_back(ScreenElement(1, "button", 100, 200, 80, 30, "Login", true));
    screen1.push_back(ScreenElement(2, "textbox", 100, 150, 150, 25, "", false));
    controller.simulateNewScreenData(screen1);

    // Run the controller's cognitive cycle.
    controller.run(1);


    // --- Simulation Run 2: Dashboard Screen ---
    std::cout << "\n--- Starting Run 2: Navigating a Dashboard ---" << std::endl;

    // Simulate a different screen state, perhaps after a successful login.
    std::vector<ScreenElement> screen2;
    screen2.push_back(ScreenElement(10, "image", 50, 50, 400, 300, "", false));
    screen2.push_back(ScreenElement(11, "button", 200, 400, 100, 40, "Submit", true));
    controller.simulateNewScreenData(screen2);

    // Run the controller for another cycle.
    controller.run(1);

    std::cout << "\n--- Simulation Complete ---" << std::endl;

    return 0;
}

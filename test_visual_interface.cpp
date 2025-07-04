#include <NeuroGen/VisualInterface.h>
#include <memory>

int main() {
    std::unique_ptr<VisualInterface> vi = std::make_unique<VisualInterface>(224, 224);
    return 0;
}

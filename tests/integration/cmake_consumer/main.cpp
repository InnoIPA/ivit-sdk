#include <iostream>
#include <ivit/ivit.hpp>

int main() {
    ivit::Tensor tensor({1, 3, 8, 8}, ivit::DataType::Float32);
    if (tensor.shape().size() != 4)
        return 1;
    std::cout << "iVIT " << ivit::version() << '\n';
    for (const auto &device : ivit::list_devices()) {
        std::cout << device.id << ' ' << device.backend << '\n';
    }
    return 0;
}

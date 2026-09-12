#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <autodiff/forward/real/real.hpp>
#include "numerical_model.hpp"

// Константа фона из статьи: A(z) = -C/8 * z^2, C = 1.16 ГэВ^2
constexpr double C_WARP = 1.16;


int main() {
    // 1. Определение деформации метрики A(z) = -C/8 * z^2
    auto deform_autodiff = [](autodiff::real3rd z) {
        return -static_cast<autodiff::real3rd>(C_WARP / 8.0) * z * z;
    };
    auto deform_num = [](double z) {
        return -(C_WARP / 8.0) * z * z;
    };

    const size_t net_nodes = 1000;
    const double z_max_val = 9.9;
    const double dummy_k   = 1.0; 
    std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium, Heavy_Mesons::Bottomonium, Heavy_Mesons::Tetraquark, Heavy_Mesons::Hybrid_Meson};

    std::cout << std::fixed << std::setprecision(4);

    for (const auto& config : targets) {
        std::cout << "====================================================\n";
        std::cout << "  Тестирование: " << config.name << "\n";
        std::cout << "  Параметры: m_bar = " << config.m_bar 
                  << " ГэВ, M = " << config.M 
                  << " ГэВ, sqrt(G) = " << config.sqrt_gamma << " ГэВ\n";
        std::cout << "====================================================\n";

        // --- Тест 1: Автодифференцирование (autodiff) ---
        auto model_a = Mesons_Model<autodiff::real3rd>(net_nodes, z_max_val, dummy_k, deform_autodiff);
        model_a.heavy_meson = config;

        Eigen::VectorXd res_a = model_a.get_exotic_meson_mass_a();

        // --- Тест 2: Численный расчет (разностная схема) ---
        auto model_num = Mesons_Model<double>(net_nodes, z_max_val, dummy_k, deform_num);
        model_num.heavy_meson = config;
        Eigen::VectorXd res_num = model_num.get_exotic_meson_mass_num();

        // Вывод результатов
        std::cout << " n | Autodiff (ГэВ) | Численно (ГэВ) | В статье (ГэВ)\n";
        std::cout << "---+---------------+----------------+----------------\n";
        for (size_t i = 0; i < config.paper_masses_gev.size(); ++i) {
            // Извлекаем квадратный корень из собственных значений (m = \sqrt{\lambda})
            double mass_a   = std::sqrt(res_a(i));
            double mass_num = std::sqrt(res_num(i));
            double mass_ref = config.paper_masses_gev[i];

            std::cout << " " << (i + 1) << " |    " 
                      << std::setw(8) << mass_a << "   |     "
                      << std::setw(8) << mass_num << "   |     "
                      << std::setw(8) << mass_ref << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}
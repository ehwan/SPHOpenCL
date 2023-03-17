#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "engine.hpp"

void render( float R, sf::RenderWindow &window, sf::Transform &transform,
    std::vector<ehfloat2> &position, std::vector<int> &flags, std::vector<ehfloat>&, std::vector<ehfloat2> &vel, std::vector<ehfloat> &rho, std::vector<int> &, std::vector<ehfloat2> &gx, std::vector<ehfloat2> &gy );

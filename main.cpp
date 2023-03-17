#include "engine.hpp"
#include "render.hpp"
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <set>
#include <fstream>
#include <chrono>
#include "karmavortex.hpp"

int windowWidth = 2000;
int windowHeight = 600;
karma_vortex_t engine;
int mode = 0;

int main()
{
  engine.load( 1.0/6.0, 1, 100, 1 );
  //engine.dt = std::min(engine.dt,1.0/200.0);
  windowWidth = engine.windowWidth;
  windowHeight = engine.windowHeight;
  sf::RenderWindow window(sf::VideoMode(windowWidth,windowHeight),"HelloWindow" );
  sf::Transform gameTransform;

  float gx = engine.engine.maxbound.s[0] - engine.engine.minbound.s[0];
  float gy = engine.engine.maxbound.s[1] - engine.engine.minbound.s[1];
  gameTransform.translate( 0, windowHeight );
  gameTransform.scale(
      (float)windowWidth/gx,
      -(float)windowHeight/gy
  );
  gameTransform.translate(
      -engine.engine.minbound.s[0],
      -engine.engine.minbound.s[1]
  );

  //gameTransform.scale( sf::Vector2f(windowWidth/engine.gameWidth,windowHeight/engine.gameHeight)*1.0f );


  int apressed = 0;
  int spressed = 0;
  int sreleased = 0;
  int stepped = 0;

  window.clear( sf::Color::White);
  auto pos = engine.engine.template get_buffer<ehfloat2>( engine.engine.position );
  auto flags = engine.engine.template get_buffer<cl_int>( engine.engine.flags );
  //render( engine.engine.gap*0.4, window, gameTransform, pos, flags );
  window.display();
  int renderi = 0;
  int renderstep = 1;
  while( window.isOpen() )
  {
    sf::Event event;
    while( window.pollEvent(event) )
    {
      if( event.type == sf::Event::Closed )
      {
        window.close();
      }
    }
    if( sf::Keyboard::isKeyPressed( sf::Keyboard::A ) )
    {
      apressed = 1;
    }else
    {
      apressed = 0;
    }
    if( sf::Keyboard::isKeyPressed( sf::Keyboard::S ) )
    {
      spressed = 1;
    }else
    {
      if( spressed )
      {
        sreleased = 1;
        spressed = 0;
      }
    }

    apressed = 1;

    if( apressed || sreleased )
    {
      sreleased = 0;
      engine.step();
      if( renderi == 0 )
      {
        renderi = renderstep;
        window.clear( sf::Color::White);
        pos = engine.engine.template get_buffer<ehfloat2>( engine.engine.position );
        flags = engine.engine.template get_buffer<cl_int>( engine.engine.flags );
        auto pressure = engine.engine.template get_buffer<ehfloat>( engine.engine.pressure );
        auto vel = engine.engine.template get_buffer<ehfloat2>( engine.engine.velocity );
        auto rho = engine.engine.template get_buffer<ehfloat>( engine.engine.rho );
        auto color = engine.engine.template get_buffer<cl_int>( engine.engine.color );
        render( engine.engine.gap*0.4, window, gameTransform, pos, flags, pressure, vel, rho, color );
        window.display();
        std::cout << engine.t << "\n";
      }
      --renderi;
    }
  }
  return 0;
}

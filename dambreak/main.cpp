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
#include "dambreak.hpp"

int windowWidth = 1200;
int windowHeight = 1200;
dambreak_t engine;

int main( int argc, char **argv )
{
  if( argc > 1 )
  {
    ehfloat H = std::atof( argv[1] );
    engine.load(H);
  }else
  {
    engine.load( 0.03 );
  }
  sf::RenderWindow window(sf::VideoMode(windowWidth,windowHeight),"HelloWindow" );
  sf::Sprite sprite;
  sprite.setOrigin(0,0);
  sprite.setScale(1.0/engine.image_width,1.0/engine.image_height);
  sf::Texture image;
  sf::Transform gameTransform;

  gameTransform.translate( 0, windowHeight );
  gameTransform.scale( (float)windowWidth, -(float)windowHeight );
  image.create(engine.image_height,engine.image_height);
  sprite.setTexture(image);

  int apressed = 0;
  int spressed = 0;
  int sreleased = 0;
  int rpressed = 0;
  int rreleased = 0;
  int stepped = 0;
  int mode = 0;

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
    if( sf::Keyboard::isKeyPressed( sf::Keyboard::R ) )
    {
      rpressed = 1;
    }else
    {
      if( rpressed )
      {
        rreleased = 1;
        ++mode;
        mode %= 2;
        rpressed = 0;
      }
    }


    if( apressed || sreleased || rreleased )
    {
      sreleased = 0;
      engine.step();
      if( renderi == 0 || rreleased )
      {
        rreleased = 0;
        renderi = renderstep;
        if( mode == 0 )
        {
          window.clear( sf::Color::Black );
          auto pos = engine.engine.template get_buffer<ehfloat2>( engine.engine.position );
          auto flags = engine.engine.template get_buffer<cl_int>( engine.engine.flags );
          auto pressure = engine.engine.template get_buffer<ehfloat>( engine.engine.pressure );
          auto vel = engine.engine.template get_buffer<ehfloat2>( engine.engine.velocity );
          auto rho = engine.engine.template get_buffer<ehfloat>( engine.engine.rho );
          auto color = engine.engine.template get_buffer<cl_int>( engine.engine.color );
          auto gradvy = engine.engine.template get_buffer<ehfloat2>( engine.engine.gradvx );
          auto gradvx = engine.engine.template get_buffer<ehfloat2>( engine.engine.gradvy );
          render( engine.engine.gap*0.4, window, gameTransform, pos, flags, pressure, vel, rho, color, gradvx, gradvy );
          window.display();
        }
        if( mode == 1 )
        {
          window.clear( sf::Color::Black );
          engine.get_image();
          image.update( engine.pixels.data() );
          window.draw( sprite, gameTransform );
          window.display();
        }
        //std::cout << engine.t << " " << engine.engine.N << "\n";
      }
      --renderi;
    }
  }
  return 0;
}

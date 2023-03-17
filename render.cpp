#include "render.hpp"
#include <cmath>
#include <iostream>
#include "flags.h"
//#include "wave.hpp"

void render( float R, sf::RenderWindow &window, sf::Transform &transform,
    std::vector<ehfloat2> &position, std::vector<int> &flags, std::vector<ehfloat> &pressure, std::vector<ehfloat2> &vel, std::vector<ehfloat> &rho,
    std::vector<int> &colors )
{
  sf::VertexArray va( sf::Quads, position.size()*4 );
  ehfloat Cs = 15;
  ehfloat gamma = 7;
  ehfloat p0 = Cs*Cs/gamma;
  ehfloat pressure_gap = 0.1;
  ehfloat pressure_mid = 1.0;
  ehfloat vel_gap = 0.4;
  int N = 0;
  for( int i=0; i<position.size(); ++i )
  {
    auto p = position[i];
    //ehfloat press = rho[i]-0.9;
    ehfloat press = (pressure[i]/p0)-pressure_mid;
    ehfloat v = std::sqrt( vel[i].s[0]*vel[i].s[0] + vel[i].s[1]*vel[i].s[1] );
    sf::Color c;
    using Uint8 = unsigned char;

    if( colors[i] == 0 )
    {
      c = sf::Color( 0x88, 0x99, 0xff );
      //c = sf::Color::Blue;
    }else
    {
      c = sf::Color( 0xff, 0x99, 0x66 );
      //c = sf::Color::Red;
    }
    va[4*N].color = c;
    va[4*N+1].color = c;
    va[4*N+2].color = c;
    va[4*N+3].color = c;

    va[4*N+0].position = { (float)p.s[0]-R, (float)p.s[1]-R };
    va[4*N+1].position = { (float)p.s[0]-R, (float)p.s[1]+R };
    va[4*N+2].position = { (float)p.s[0]+R, (float)p.s[1]+R };
    va[4*N+3].position = { (float)p.s[0]+R, (float)p.s[1]-R };
    ++N;
  }
  va.resize( 4*N );
  window.draw( va, transform );
}

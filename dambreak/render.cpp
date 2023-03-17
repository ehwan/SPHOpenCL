#include "render.hpp"
#include <cmath>
#include <iostream>
#include "flags.h"
//#include "wave.hpp"

void render( float R, sf::RenderWindow &window, sf::Transform &transform,
    std::vector<ehfloat2> &position, std::vector<int> &flags, std::vector<ehfloat> &pressure, std::vector<ehfloat2> &vel, std::vector<ehfloat> &rho,
    std::vector<int> &colors, std::vector<ehfloat2> &gx, std::vector<ehfloat2> &gy )
{
  sf::VertexArray va( sf::Quads, position.size()*4 );
  int N = 0;
  for( int i=0; i<position.size(); ++i )
  {
    //if( (flags[i] & EH_PARTICLE_PLOT) == 0 ){ continue; }
    auto p = position[i];
    sf::Color c;
    using Uint8 = unsigned char;

    ehfloat v = std::sqrt(vel[i].s[0]*vel[i].s[0] + vel[i].s[1]*vel[i].s[1]);
    ehfloat factor = 0.8 + v*0.2;
    ehfloat r = (ehfloat)0x99/255.0;
    ehfloat g = (ehfloat)0xcc/255.0;
    ehfloat b = (ehfloat)0xff/255.0;
    r *= factor;
    g *= factor;
    b *= factor;
    r = std::min(r,1.0f);
    g = std::min(g,1.0f);
    b = std::min(b,1.0f);

    c = sf::Color( (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255) );

    /*
    if( colors[i] == 0 )
    {
      c = sf::Color::Black;
    }else
    {
      c = sf::Color( 0x99, 0xcc, 0xff );
    }
    */
    /*
    if( press > 0 )
    {
      ehfloat gb = 1.0 - press/pressure_gap;
      c = sf::Color( 255, (Uint8)(255*gb), (Uint8)(255*gb) );
    }else
    {
      ehfloat rg = 1.0 + press/pressure_gap;
      c = sf::Color( (Uint8)(255*rg), (Uint8)(255*rg), 255 );
    }
    */

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

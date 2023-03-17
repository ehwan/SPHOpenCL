#include "engine.hpp"


struct dambreak_t
{
  engine_t engine;
  ehfloat t = 0;
  std::uniform_real_distribution<ehfloat> dist{ -1, 1 };
  std::mt19937 mt{ std::random_device{}() };

  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&
  > set_wall_velocity_kernel{ cl::Kernel() };

  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl_int,cl_int
  > get_image_kernel{ cl::Kernel() };
  cl::Buffer image_buffer;
  std::vector<unsigned char> pixels;
  int image_width = 1000;
  int image_height = 1000;
  int image_global_width = 1024;
  int image_global_height = 1024;


  void load( ehfloat H )
  {
    param_t param;
    param.gravity = { 0, -9 };
    param.Cs = 10;
    param.eta = 3.0;
    param.h = H;
    param.max_particle_count = (1.0/(param.h/param.eta)*1.0/(param.h/param.eta) * 1.3);
    param.minbound = { -1, -3*param.h };
    param.maxbound = { 2, 2 };
    param.gamma = 7;
    param.courant_dt_factor = 1.0;
    param.diffusion_dt_factor = 100;
    param.mu = 0.01;
    param.rho0 = 1;
    
    engine.set( param );
    engine.load_opencl();
    set_wall_velocity_kernel =
      decltype(set_wall_velocity_kernel)( engine.program, "set_wall_velocity" );
    get_image_kernel =
      decltype(get_image_kernel)( engine.program, "get_image" );
    image_buffer = cl::Buffer( engine.context, CL_MEM_READ_WRITE, sizeof(cl_uchar)*4*image_width*image_height );
    pixels.resize( image_width*image_height*4 );

    // wall particles
    for( int yi=0; yi<7; ++yi )
    {
      for( ehfloat x=0.5*engine.gap; x<1; x+=engine.gap )
      {
        ehfloat y = -yi*engine.gap;
        particle_info_t info;
        info.position = {x,y};
        info.velocity = {0,0};
        info.svelocity = {0,0};
        info.flag = 
          0
          |EH_PARTICLE_STATIC
          |EH_PARTICLE_STATICMOVE
          |EH_PARTICLE_NOFORCE
          |EH_PARTICLE_PRESSURECLAMP
          |EH_PARTICLE_WALL;
        info.color = 0;

        engine.add_particle( info );
      }
    }

    // fluid particles
    for( ehfloat y=engine.gap; y<0.8; y+=engine.gap )
    {
      for( ehfloat x=0.1; x<0.43; x+=engine.gap )
      {
        particle_info_t info;
        ehfloat rx = dist(mt)*engine.gap*0.1;
        ehfloat ry = dist(mt)*engine.gap*0.1;
        info.position = { x+rx, y+ry };
        info.velocity = {0,0};
        info.svelocity = { 0,0};
        info.flag =
          0
          |EH_PARTICLE_PRESSURECLAMP
          |EH_PARTICLE_FLUID;
        info.color = 1;
        engine.add_particle( info );
      }
    }
    engine.add_waitlist();
  }

  void set_wall_velocity()
  {
    cl_int err;
    set_wall_velocity_kernel(
        cl::EnqueueArgs(engine.queue,cl::NDRange(engine.global_work_size)),
        engine.constant_buffer,
        engine.grid_particlecount,
        engine.position,
        engine.velocity,
        engine.rho,
        engine.flags,
        engine.V,
        err
    ).wait();
    engine.check_kernel_error( err, "set_wall_velocity" );
  }
  void get_image()
  {
    cl_int err;
    get_image_kernel(
        cl::EnqueueArgs(engine.queue,cl::NDRange(image_global_width,image_global_height)),
        engine.constant_buffer,
        engine.grid_particlecount,
        engine.position,
        engine.velocity,
        engine.rho,
        engine.flags,
        engine.V,
        image_buffer,
        image_width,image_height,
        err
    ).wait();
    engine.check_kernel_error( err, "get_image" );
    engine.queue.enqueueReadBuffer( image_buffer, CL_TRUE, 0, image_width*image_height*4, pixels.data() );
  }

  void step()
  {
    t += engine.dt;
    engine.add_waitlist();
    engine.calculate_global_work_size();
    engine.upload_constants();
    engine.queue.flush();
    engine.queue.finish();

    engine.grid_sort();
    engine.calculate_rho();
    set_wall_velocity();
    engine.calculate_nonpressure_force();
    engine.advect_phase1();
    engine.queue.flush();
    engine.queue.finish();

    engine.grid_sort();
    engine.calculate_rho();
    engine.calculate_pressure();
    engine.calculate_pressure_force();
    engine.advect_phase2();
  }
};

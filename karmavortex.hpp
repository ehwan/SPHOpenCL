#pragma once

#include "engine.hpp"
#include <random>
#include <iostream>
#include <stdexcept>

struct karma_vortex_t
{
  engine_t engine;

  ehfloat gameWidth = 10;
  ehfloat gameHeight = 3.0;
  int windowWidth = 2000;
  int windowHeight = 1000;

  int force_div = 64;
  ehfloat t = 0;

  ehfloat H;
  ehfloat gap;

  ehfloat Re;
  ehfloat U;
  ehfloat D;

  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&
  > leftright_kernel{ cl::Kernel() };
  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&
  > calculate_cylinder_force_kernel{ cl::Kernel() };
  cl::Buffer force_buffer;

  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl_int
  > get_velocity_kernel{ cl::Kernel() };
  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&
  > set_cylinder_velocity_kernel{ cl::Kernel() };
  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&
  > set_updown_velocity_kernel{ cl::Kernel() };
  cl::KernelFunctor<
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl::Buffer&,
    cl_int,cl_int,ehfloat
  > get_image_kernel{ cl::Kernel() };
  cl::Buffer image_buffer;
  cl::Buffer image_pos_buffer;




  struct Nbuffer_t
  {
    ehfloat U, D;
    cl_int dummyN;
    cl_int N;
    cl_int left_boundary_flag;
    cl_int right_boundary_flag;
  };
  struct plot_data_t
  {
    ehfloat2 point;
    ehfloat2 velocity;
    ehfloat2 point2;
    ehfloat2 velocity2;
    ehfloat2 viscous;
    ehfloat2 pressure;
    int N = 2;
  };

  // buffer for calculation
  cl::Buffer plot_buffer;
  plot_data_t plot_data;
  cl::Buffer Nbuffer;
  Nbuffer_t Nbuffer_data;

  // U * dt < h/eta
  // dt < h/eta / U

  void load( ehfloat H_, ehfloat D_, ehfloat Re_, ehfloat U_ )
  {
    H = H_;
    D = D_;
    U = U_;
    Re = Re_;

    gameWidth = 20*D;
    gameHeight = 10*D;

    param_t params;
    params.rho0 = 1;
    params.gamma = 7;
    params.eta = 3.333333333333;
    params.max_particle_count = (int)(gameHeight*D/(H/params.eta)*gameWidth*D/(H/params.eta)*1.4);
    params.h = H;
    params.mu = D*U*params.rho0/Re;
    params.Cs = 10*U;
    params.minbound = {-(ehfloat)5*H,-(ehfloat)5*H};
    params.maxbound = {gameWidth+(ehfloat)5*H,gameHeight+(ehfloat)5*H};
    params.courant_dt_factor = 1.0;
    params.diffusion_dt_factor = 0.5;
    //params.local_size = 128;
    engine.set( params );
    engine.mass = engine.rho0*(engine.gap*engine.gap);
    engine.load_opencl();
    get_velocity_kernel =
      decltype(get_velocity_kernel)( engine.program, "get_velocity" );
    leftright_kernel = 
      decltype(leftright_kernel)( engine.program, "karman_leftright" );
    calculate_cylinder_force_kernel =
      decltype( calculate_cylinder_force_kernel )( engine.program, "calculate_cylinder_force" );
    set_cylinder_velocity_kernel =
      decltype( set_cylinder_velocity_kernel )( engine.program, "set_cylinder_velocity" );
    set_updown_velocity_kernel =
      decltype( set_updown_velocity_kernel )( engine.program, "set_updown_velocity" );
    get_image_kernel =
      decltype( get_image_kernel )( engine.program, "get_image" );
    image_buffer = cl::Buffer( engine.context, CL_MEM_READ_WRITE, sizeof(ehfloat2)*1000*500 );
    image_pos_buffer = cl::Buffer( engine.context, CL_MEM_READ_WRITE, sizeof(ehfloat2)*1000*500 );

    Nbuffer = cl::Buffer( engine.context, CL_MEM_READ_WRITE, sizeof(Nbuffer_t) );
    Nbuffer_data.U = U;
    Nbuffer_data.D = D;
    plot_buffer = cl::Buffer( engine.context, CL_MEM_READ_WRITE, sizeof(plot_data) );
    force_buffer = cl::Buffer( engine.context, CL_MEM_READ_WRITE, sizeof(ehfloat2)*2*128 );

    gap = engine.gap;

    std::cout << "U : " << U << "\n";
    std::cout << "D : " << D << "\n";
    std::cout << "Reynolds : " << Re << "\n";

    int left_boundary_flag =
            0
            |EH_PARTICLE_STATIC
            |EH_PARTICLE_LEFT
            //|EH_PARTICLE_PRESSURECLAMP;
            |EH_PARTICLE_STATICMOVE
            |EH_PARTICLE_NOFORCE;
    int right_boundary_flag =
            0;
            //|EH_PARTICLE_STATIC
            //|EH_PARTICLE_RIGHT
            //|EH_PARTICLE_PRESSURECLAMP;
            //|EH_PARTICLE_STATICMOVE
            //|EH_PARTICLE_NOFORCE;
    Nbuffer_data.left_boundary_flag = left_boundary_flag;
    Nbuffer_data.right_boundary_flag = right_boundary_flag;

    int yN = 10/engine.gap;
    int xN = 20/engine.gap;
    int dummyN = 7;
    Nbuffer_data.dummyN = dummyN;
    for( int yi=-dummyN; yi<=yN+dummyN; ++yi )
    {
      for( int xi=-dummyN; xi<=xN; ++xi )
      {
        ehfloat x = xi*engine.gap;
        ehfloat y = yi*engine.gap;
        ehfloat dx = x-5;
        ehfloat dy = y-5;
        ehfloat rsq = dx*dx+dy*dy;
        ehfloat rmin = 0.5;
        ehfloat rmax = 0.5+engine.gap;
        if( rsq < 0.5*0.5 )
        {
          // cylinder dummy
          particle_info_t par;
          par.flag = 
            EH_PARTICLE_STATIC
            |EH_PARTICLE_CYLINDER
            |EH_PARTICLE_STATICMOVE
            |EH_PARTICLE_NOFORCE;
          par.position = { x,y };
          par.velocity = {0,0};
          par.svelocity = {0,0};
          engine.add_particle( par );
        }else if( yi < 0 )
        {
          // down
          particle_info_t par;
          par.flag =
            EH_PARTICLE_STATIC
            |EH_PARTICLE_DOWN
            |EH_PARTICLE_STATICMOVE
            |EH_PARTICLE_NOFORCE;
          par.position = { x,y };
          par.velocity = {U,0};
          par.svelocity = {0,0};
          engine.add_particle( par );
        }else if( yi > yN )
        {
          // up
          particle_info_t par;
          par.flag =
            EH_PARTICLE_STATIC
            |EH_PARTICLE_UP
            |EH_PARTICLE_STATICMOVE
            |EH_PARTICLE_NOFORCE;
          par.position = { x,y };
          par.velocity = {U,0};
          par.svelocity = {0,0};
          engine.add_particle( par );
        }else if( xi < 0 )
        {
          particle_info_t par;
          par.flag =
            left_boundary_flag;
          par.position = { x,y };
          par.velocity = {U,0};
          par.svelocity = {U,0};
          engine.add_particle( par );
        }else
        {
          particle_info_t par;
          par.flag = 0;
          par.position = { x,y };
          par.velocity = {U,0};
          par.svelocity = {U,0};
          engine.add_particle( par );
        }
      }
    }
    engine.add_waitlist();
  }

  void leftright()
  {
    engine.add_waitlist();
    engine.calculate_global_work_size();
    engine.upload_constants();
    cl_int err;
    Nbuffer_data.N = engine.N;
    engine.queue.enqueueWriteBuffer( Nbuffer, CL_TRUE, 0, sizeof(Nbuffer_t), &Nbuffer_data );
    leftright_kernel(
        cl::EnqueueArgs(engine.queue,cl::NDRange(engine.global_work_size)),
        engine.constant_buffer,
        engine.position,
        engine.velocity,
        engine.svelocity,
        engine.flags,
        engine.color,
        Nbuffer,
        err
        ).wait();
    engine.check_kernel_error( err, "error leftright" );
    engine.queue.enqueueReadBuffer( Nbuffer, CL_TRUE, 0, sizeof(Nbuffer_t), &Nbuffer_data );
    engine.N = Nbuffer_data.N;
  }
  void calculate_cylinder_force()
  {
    engine.queue.enqueueWriteBuffer(plot_buffer,CL_TRUE,0,sizeof(plot_data),&plot_data);
    cl_int err;
    calculate_cylinder_force_kernel(
        cl::EnqueueArgs(engine.queue,cl::NDRange(force_div)),
        engine.constant_buffer,
        engine.grid_particlecount,
        engine.position,
        engine.velocity,
        engine.flags,
        engine.rho,
        engine.V,
        engine.pressure,
        force_buffer,
        plot_buffer,
        err
    ).wait();
    engine.check_kernel_error( err, "error calculate_cylinder_force" );
    engine.queue.enqueueReadBuffer( plot_buffer, CL_TRUE, 0, sizeof(plot_data), &plot_data );
  }
  void get_velocity()
  {
    engine.queue.enqueueWriteBuffer(plot_buffer,CL_TRUE,0,sizeof(plot_data),&plot_data);
    cl_int err;
    get_velocity_kernel(
        cl::EnqueueArgs(engine.queue,cl::NDRange(plot_data.N)),
        engine.constant_buffer,
        engine.grid_particlecount,
        engine.position,
        engine.velocity,
        engine.rho,
        engine.flags,
        engine.V,
        plot_buffer,
        plot_data.N,
        err
    ).wait();
    engine.check_kernel_error( err, "error get_velocity" );
    engine.queue.enqueueReadBuffer( plot_buffer, CL_TRUE, 0, sizeof(plot_data), &plot_data );
  }
  void set_cylinder_velocity()
  {
    cl_int err;
    set_cylinder_velocity_kernel(
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
    engine.check_kernel_error( err, "set_cylinder_velocity" );
  }
  void set_updown_velocity()
  {
    cl_int err;
    set_updown_velocity_kernel(
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
    engine.check_kernel_error( err, "set_updown_velocity" );
  }
  void phase1()
  {
    leftright();

    engine.add_waitlist();
    engine.calculate_global_work_size();
    engine.upload_constants();
    engine.queue.flush();
    engine.queue.finish();

    engine.grid_sort();
    engine.calculate_rho();
    engine.calculate_pressure();
    set_cylinder_velocity();
    set_updown_velocity();
    engine.calculate_nonpressure_force();
  }
  void phase2()
  {
    engine.advect_phase1();
    engine.queue.flush();
    engine.queue.finish();

    engine.grid_sort();
    engine.calculate_rho();
    engine.calculate_pressure();
    engine.calculate_pressure_force();
    calculate_cylinder_force();
  }
  void phase3()
  {
    engine.advect_phase2();

    get_velocity();
    t += engine.dt;
  }
  std::pair<std::vector<ehfloat2>,std::vector<ehfloat2>> get_image( int W, int H )
  {
    cl_int err;
    get_image_kernel(
        cl::EnqueueArgs(engine.queue,cl::NDRange(1200,600)),
        engine.constant_buffer,
        engine.grid_particlecount,
        engine.position,
        engine.velocity,
        engine.rho,
        engine.flags,
        engine.V,
        image_buffer,
        image_pos_buffer,
        W, H, H*0.8,
        err
    ).wait();
    engine.check_kernel_error( err, "error get_image" );

    std::vector<ehfloat2> poss( W*H );
    std::vector<ehfloat2> vels( W*H );
    engine.queue.enqueueReadBuffer( image_pos_buffer, CL_TRUE, 0, sizeof(ehfloat2)*W*H, poss.data() );
    engine.queue.enqueueReadBuffer( image_buffer, CL_TRUE, 0, sizeof(ehfloat2)*W*H, vels.data() );
    return { poss, vels };
  }
  void step()
  {
    leftright();

    engine.add_waitlist();
    engine.calculate_global_work_size();
    engine.upload_constants();
    engine.queue.flush();
    engine.queue.finish();

    engine.grid_sort();
    engine.calculate_rho();
    set_cylinder_velocity();
    set_updown_velocity();
    engine.calculate_nonpressure_force();
    engine.advect_phase1();
    engine.queue.flush();
    engine.queue.finish();

    engine.grid_sort();
    engine.calculate_rho();
    engine.calculate_pressure();
    engine.calculate_pressure_force();
    //calculate_cylinder_force();
    engine.advect_phase2();

    get_velocity();
    t += engine.dt;
  }
};

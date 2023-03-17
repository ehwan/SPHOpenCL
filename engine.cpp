#include "engine.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

void engine_t::load_opencl()
{
  if( debug )
  {
    std::cout << "~~~~~OpenCL Initialize~~~~~~~\n";
  }
  int maxN = max_particle_count;
  platform = cl::Platform::get();
  //cl::Platform::setDefault( platform );

  {
    std::vector<cl::Device> devices;
    platform.getDevices( CL_DEVICE_TYPE_GPU, &devices );
    device = std::move( devices[0] );
  }
  //cl::Device::setDefault( device );

  context = cl::Context( device );
  //cl::Context::setDefault( context );
  std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
  double_support = extensions.find(std::string("cl_khr_fp64")) != std::string::npos;
  if( debug )
  {
    std::cout << "Platform : " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "Device : " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "Max Compute Unit : " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    //std::cout << extensions << "\n";
    std::cout << "double support : " << double_support << "\n";
  }
#if USE_DOUBLE==1
  if( double_support == false )
  {
    std::cout << "double not supported but using ehfloat=double;\n";
    throw std::runtime_error( "double not supported error" );
  }

  if( debug )
  {
    std::cout << "using double precision;\n";
  }
#else
  if( debug )
  {
    std::cout << "using float precision;\n";
  }
#endif

  // buffers
  constant_buffer = cl::Buffer( context, CL_MEM_READ_ONLY, sizeof(constant_t) );

  position = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );
  position_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );

  velocity = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );
  velocity_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );

  svelocity = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );
  svelocity_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );

  flags = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(cl_int) );
  flags_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(cl_int) );

  rho = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat) );
  V = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat) );

  pressure = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat) );
  color = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(cl_int) );
  color_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(cl_int) );

  omega = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat) );
  omega_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat) );
  gradvx = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );
  gradvx_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );
  gradvy = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );
  gradvy_pong = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );

  nonpressure_force = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );
  pressure_force = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(ehfloat2) );

  grid_localindex = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(cl_int) );
  gridindex = cl::Buffer( context, CL_MEM_READ_WRITE, maxN*sizeof(cl_int) );

  grid_particlecount = cl::Buffer( context, CL_MEM_READ_WRITE, (gridsize1+1)*sizeof(cl_int) );


  queue = cl::CommandQueue( context, device );
  //cl::CommandQueue::setDefault( queue );


  // loading sources
  std::ifstream file( "kernels.cl" );
  file.seekg( 0, std::ifstream::end );
  size_t filesize = file.tellg();
  std::string source;
  source.resize( filesize );
  file.seekg( 0, std::ifstream::beg );
  file.read( &source[0], filesize );

#if USE_DOUBLE==1
  std::string typedef_ehfloat =
    "typedef double ehfloat;\n"
    "typedef double2 ehfloat2;\n"
    "typedef double3 ehfloat3;\n"
    "typedef double4 ehfloat4;\n"
    "typedef double8 ehfloat8;\n"
    "typedef double16 ehfloat16;\n";
  std::string build_options =
    "-cl-std=CL1.2 -D EH_PI=M_PI";
#else
  std::string typedef_ehfloat =
    "typedef float ehfloat;\n"
    "typedef float2 ehfloat2;\n"
    "typedef float3 ehfloat3;\n"
    "typedef float4 ehfloat4;\n"
    "typedef float8 ehfloat8;\n"
    "typedef float16 ehfloat16;\n";
  std::string build_options =
    "-cl-std=CL1.2 -D EH_PI=M_PI_F";
#endif

  cl::Program::Sources sources;
  sources.push_back( {typedef_ehfloat.c_str(), typedef_ehfloat.size()} );
  sources.push_back( {source.c_str(), source.size()} );

  program = cl::Program( context, sources );
  if( program.build( {device}, build_options.c_str() ) != CL_SUCCESS )
  {
    std::cout << "error building kernel\n";
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second << "\n";
    throw std::runtime_error( "program building error" );
  }

  kernels.assume_grid_count =
    decltype(kernels.assume_grid_count)( program, "assume_grid_count" );
  kernels.gridcount_accumulate =
    decltype(kernels.gridcount_accumulate)( program, "gridcount_accumulate" );
  kernels.move_to_new_grid =
    decltype(kernels.move_to_new_grid)( program, "move_to_new_grid" );
  kernels.calculate_nonpressure_force =
    decltype(kernels.calculate_nonpressure_force)( program, "calculate_nonpressure_force" );
  kernels.calculate_pressure =
    decltype(kernels.calculate_pressure)( program, "calculate_pressure" );
  kernels.calculate_pressure_force =
    decltype(kernels.calculate_pressure_force)( program, "calculate_pressure_force" );
  kernels.advect_phase1 =
    decltype(kernels.advect_phase1)( program, "advect_phase1" );
  kernels.advect_phase2 =
    decltype(kernels.advect_phase2)( program, "advect_phase2" );
  kernels.calculate_rho =
    decltype(kernels.calculate_rho)( program, "calculate_rho" );


  upload_constants();
}
void engine_t::set( param_t &param )
{
  max_particle_count = param.max_particle_count;
  N = 0;
  H = param.h;
  invH = 1.0/param.h;
  mu = param.mu;
  mass = 1;
  eta = param.eta;
  gap = H/eta;
  minbound = param.minbound;
  maxbound = param.maxbound;
  gridsize.s[0] = (int)std::ceil((maxbound.s[0]-minbound.s[0])*invH);
  gridsize.s[1] = (int)std::ceil((maxbound.s[1]-minbound.s[1])*invH);
  gridsize1 = gridsize.s[0]*gridsize.s[1];
  if( debug )
  {
    std::cout << "maxparticle : " << max_particle_count << "\n";
    std::cout << "bound : (" << minbound.s[0] << ", " << minbound.s[1] << ")x("
      << maxbound.s[0] << ", " << maxbound.s[1] << ")\n";
    std::cout << "gridsize : (" << gridsize.s[0] << ", " << gridsize.s[1] << ")\n";
  }

  rho0 = param.rho0;
  Cs = param.Cs;
  gamma = param.gamma;

  ehfloat courant_dt = param.courant_dt_factor*gap/Cs;
  ehfloat diffusion_dt = 100000;
  if( mu > 0 )
  {
    diffusion_dt = param.diffusion_dt_factor*0.25*gap*gap/mu;
  }
  dt = std::min( courant_dt, diffusion_dt );
  gravity = param.gravity;
  pressure0 = Cs*Cs*rho0/gamma;
  mass = rho0*gap*gap;

  if( debug )
  {
    this->log();
  }
}
void engine_t::log()
{
  std::cout << "H : " << H << "\n";
  std::cout << "mu : " << mu << "\n";
  std::cout << "rho0 : " << rho0 << "\n";
  std::cout << "gamma : " << gamma << "\n";
  std::cout << "dt : " << dt << " , " << 1.0/dt << "\n";
  std::cout << "Cs : " << Cs << "\n";
  std::cout << "pressure0 : " << Cs*Cs*rho0/gamma << "\n";
  std::cout << "mass : " << mass << "\n";
}


void engine_t::grid_sort()
{
  cl::Event event;
  queue.enqueueFillBuffer( grid_particlecount, cl_int(0), 0, sizeof(cl_int)*(gridsize1+1), 0, &event );
  event.wait();
  
  cl_int err;
  kernels.assume_grid_count(
      cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
      constant_buffer,
      grid_particlecount,
      grid_localindex,
      position,
      gridindex,
      err
  ).wait();
  check_kernel_error( err, "error assume_grid_count" );

  kernels.gridcount_accumulate(
      cl::EnqueueArgs(queue,cl::NDRange(1)),
      constant_buffer,
      grid_particlecount,
      err
  ).wait();
  check_kernel_error( err, "error gridcount_accumulate" );

  // type 0 : int
  // type 1 : ehfloat
  // type 2 : ehfloat2
  auto move_to_new_grid =
    [&]( cl::Buffer& A, cl::Buffer &newA, int type )
    {
      cl_int err;
      kernels.move_to_new_grid(
        cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
        constant_buffer,
        grid_particlecount,
        grid_localindex,
        gridindex,
        A,
        newA,
        type,
        err
      );
      check_kernel_error( err, "error move_to_new_grid" );
      std::swap( A, newA );
    };
  move_to_new_grid( position, position_pong, 2 );
  move_to_new_grid( velocity, velocity_pong, 2 );
  move_to_new_grid( svelocity, svelocity_pong, 2 );
  move_to_new_grid( flags, flags_pong, 0 );
  move_to_new_grid( gradvx, gradvx_pong, 2 );
  move_to_new_grid( gradvy, gradvy_pong, 2 );
  move_to_new_grid( omega, omega_pong, 1 );
  move_to_new_grid( color, color_pong, 0 );

  queue.enqueueReadBuffer( grid_particlecount, CL_TRUE, sizeof(cl_int)*gridsize1, sizeof(cl_int), &N );
  upload_constants();
  calculate_global_work_size();
  queue.flush();
  queue.finish();
}
void engine_t::calculate_pressure()
{
  cl_int err;
  kernels.calculate_pressure(
      cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
      constant_buffer,
      rho,
      flags,
      position,
      pressure,
      err
  ).wait();
  check_kernel_error( err, "error calculate_pressure" );
}
void engine_t::calculate_pressure_force()
{
  cl_int err;
  kernels.calculate_pressure_force(
      cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
      constant_buffer,
      grid_particlecount,
      position,
      rho,
      pressure,
      flags,
      pressure_force,
      V,
      err
  ).wait();
  check_kernel_error( err, "error calculate_pressure_force" );
}
void engine_t::advect_phase1()
{
  cl_int err;
  kernels.advect_phase1(
      cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
      constant_buffer,
      flags,
      svelocity,
      position,
      velocity,
      rho,
      nonpressure_force,
      err
  ).wait();
  check_kernel_error( err, "error advect_phase1" );
}
void engine_t::advect_phase2()
{
  cl_int err;
  kernels.advect_phase2(
      cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
      constant_buffer,
      flags,
      svelocity,

      position,
      velocity,
      rho,
      pressure_force,
      err
  ).wait();
  check_kernel_error( err, "error advect_phase2" );
}
void engine_t::calculate_rho()
{
  cl_int err;
  kernels.calculate_rho(
      cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
      constant_buffer,
      grid_particlecount,
      position,
      rho,
      V,
      flags,
      err
  ).wait();
  check_kernel_error( err, "error calculate_rho" );
}
void engine_t::calculate_mass()
{
  add_waitlist();
  mass = 1;
  upload_constants();
  calculate_global_work_size();
  grid_sort();
  calculate_rho();
  auto rs = get_buffer<ehfloat>( rho );
  auto fs = get_buffer<cl_int>( flags );
  auto xs = get_buffer<ehfloat2>( position );
  ehfloat rhomax = -1;
  for( int i=0; i<N; ++i )
  {
    if( fs[i] == 0 && xs[i].s[0]>10 )
    {
      rhomax = std::max(rhomax,rs[i]);
    }
  }
  mass = rho0/rhomax;
  if(debug)
  {
    std::cout << "calculate mass : " << mass << "\n";
  }
}
void engine_t::calculate_nonpressure_force()
{
  cl_int err;
  kernels.calculate_nonpressure_force(
      cl::EnqueueArgs(queue,cl::NDRange(global_work_size)),
      constant_buffer,
      grid_particlecount,
      position,
      rho,
      velocity,
      flags,
      omega,
      gradvx,
      gradvy,
      nonpressure_force,
      V,
      err
  ).wait();
  check_kernel_error( err, "error calculate_nonpressure_force" );
}
void engine_t::step()
{
  add_waitlist();
  calculate_global_work_size();
  upload_constants();
  queue.flush();
  queue.finish();

  grid_sort();
  calculate_rho();
  calculate_nonpressure_force();
  advect_phase1();
  queue.flush();
  queue.finish();

  grid_sort();
  calculate_rho();
  calculate_pressure();
  calculate_pressure_force();
  advect_phase2();
}

#include "flags.h"

// 定数
struct constant_t {
  ehfloat2 minbound;
  ehfloat2 maxbound;
  ehfloat2 gravity;
  int2 gridsize;
  ehfloat eta;
  ehfloat gap;
  ehfloat H;
  ehfloat invH;
  ehfloat mu;
  ehfloat mass;
  ehfloat dt;
  ehfloat Cs;
  ehfloat rho0;
  ehfloat gamma;
  ehfloat pressure0;
  int N;
  int gridsize1;
};

int2 gridindex2_from_p2( constant struct constant_t *c, ehfloat2 p )
{
  return convert_int2_rtn( (p-c->minbound)*c->invH );
}
int gridindex_from_index2( constant struct constant_t *c, int2 i2 )
{
  return i2.y*c->gridsize.x + i2.x;
}
bool out_of_bound( constant struct constant_t *c, int2 i2 )
{
  return any(i2<0) || any(i2>=c->gridsize);
}

// poly6 kernel
// W = W0 * ( 1 - (r/h)^2 )^3
ehfloat kernel_function( ehfloat invh, ehfloat2 x )
{
  ehfloat q = 1.0 - dot(x,x)*invh*invh;
  return 4.0/EH_PI * invh*invh * q*q*q;
}
ehfloat2 kernel_gradient( ehfloat invh, ehfloat2 x )
{
  ehfloat q = 1.0 - dot(x,x)*invh*invh;
  return -24.0/EH_PI *invh*invh*invh*invh * q*q * x;
}

// グリッドセルに入る粒子数カウント
kernel void assume_grid_count(
  constant struct constant_t *c,
  global int *gridcount,
  global int *grid_localindex,
  global const ehfloat2 *position,
  global int *gridindex
  )
{
  const int id = get_global_id(0);
  if( id >= c->N ){ return; }

  int2 index2 = gridindex2_from_p2( c, position[id] );
  int index1 = gridindex_from_index2( c, index2 );
  if( out_of_bound(c,index2) )
  {
    index1 = c->gridsize1;
  }
  if( any(position[id]<c->minbound) || any(position[id]>c->maxbound) )
  {
    index1 = c->gridsize1;
  }
  gridindex[id] = index1;

  grid_localindex[id] = atomic_inc( gridcount+index1 );
}
kernel void gridcount_accumulate(
  constant struct constant_t *c,
  global int *gridcount
)
{
  int count_i_minus_one = gridcount[0];
  int gsize = c->gridsize1;
  gridcount[0] = 0;
  for( int i=1; i<=gsize; ++i )
  {
    int x = gridcount[i];
    gridcount[i] = gridcount[i-1] + count_i_minus_one;
    count_i_minus_one = x;
  }
}

// 新しいグリッドに移動 ( ソートする )
kernel void move_to_new_grid(
  constant struct constant_t *c,
  global const int *grid_beginpoint,
  global const int *grid_localindex,
  global const int *gridindex,
  global const int *A,
  global int *newA,
  int type
)
{
  const int id = get_global_id(0);
  if( id >= c->N ){ return; }

  int to_id = grid_beginpoint[gridindex[id]] + grid_localindex[id];
  if( type == 0 )
  {
    newA[to_id] = A[id];
  }else if( type == 1 )
  {
    global ehfloat *newA_ = (global ehfloat*)newA;
    global const ehfloat *A_ = (global const ehfloat*)A;
    newA_[to_id] = A_[id];
  }else if( type == 2 )
  {
    global ehfloat2 *newA_ = (global ehfloat2*)newA;
    global const ehfloat2 *A_ = (global const ehfloat2*)A;
    newA_[to_id] = A_[id];
  }
}


// 密度計算
kernel void calculate_rho(
  constant struct constant_t *c,
  global const int *grid_beginpoint,

  global const ehfloat2 *position,
  global ehfloat *rho,
  global ehfloat *V,
  global const int *flags
)
{
  const int id = get_global_id(0);
  if( id >= c->N ){ return; }
  int2 index2 = gridindex2_from_p2( c, position[id] );
  int2 mingrid = max( index2-1, 0 );
  int2 maxgrid = min( index2+1, c->gridsize-1 );

  ehfloat density = 0;
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int j=begin; j<end; ++j )
    {
      ehfloat2 rij = position[id] - position[j];
      if( dot(rij,rij) > c->H*c->H ){ continue; }
      ehfloat k = kernel_function(c->invH,rij);
      density += k*c->mass;
    }
  }
  rho[id] = density;
  V[id] = c->mass/density;
}

// 勾配修正テンソル
ehfloat4 gradient_tensor(
  constant struct constant_t *c,
  global const int *grid_beginpoint,

  global const ehfloat2 *position,
  global const ehfloat *rho,
  global const int *flags,
  global const ehfloat *V,
  ehfloat2 point,
  int except_flag
)
{
  int2 index2 = gridindex2_from_p2( c, point );
  int2 mingrid = max( index2-1, 0 );
  int2 maxgrid = min( index2+1, c->gridsize-1 );

  ehfloat4 invB = (ehfloat4)( 0.0, 0.0, 0.0, 0.0 );
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int j=begin; j<end; ++j )
    {
      if( flags[j] & except_flag ){ continue; }
      ehfloat2 rij = point - position[j];
      if( dot(rij,rij) > c->H*c->H ){ continue; }
      ehfloat2 kdV = kernel_gradient( c->invH, rij ) * V[j];

      invB += (ehfloat4)( rij.x*kdV, rij.y*kdV );
    }
  }

  ehfloat det = invB[0]*invB[3] - invB[1]*invB[2];
  if( fabs(det) < GRADIENT_TENSOR_EPS )
  {
    return (ehfloat4)( 1.0, 0.0, 0.0, 1.0 );
  }else
  {
    return -(1.0/det) * (ehfloat4)( invB.w, -invB.y, -invB.z, invB.x );
  }
}

//　ラプラシアン修正テンソル
ehfloat4 laplacian_tensor(
  constant struct constant_t *c,
  global const int *grid_beginpoint,

  global const ehfloat2 *position,
  global const ehfloat *rho,
  global const int *flags,
  global const ehfloat *V,
  ehfloat2 point,
  ehfloat4 B,
  int except_id,
  int except_flag
)
{
  int2 index2 = gridindex2_from_p2( c, point );
  int2 mingrid = max( index2-1, 0 );
  int2 maxgrid = min( index2+1, c->gridsize-1 );

  ehfloat3 M[3];
  {
  ehfloat16 X = (ehfloat16)(0);
  ehfloat8 X1 = (ehfloat8)(0);
  ehfloat8 X2 = (ehfloat8)(0);
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int n=begin; n<end; ++n )
    {
      if( n == except_id ){ continue; }
      if( flags[n] & except_flag ){ continue; }
      ehfloat2 eij = point - position[n];
      if( dot(eij,eij) > c->H*c->H ){ continue; }
      //if( dot(eij,eij) < DISTANCE_EPS_SQ ){ continue; }
      ehfloat2 kdV = kernel_gradient( c->invH, eij ) * V[n];
      ehfloat r = length(eij);
      eij = normalize(eij);

      ehfloat4 t4 = (ehfloat4)( eij.x*kdV, eij.y*kdV );
      ehfloat8 t8 = (ehfloat8)( eij.x*t4, eij.y*t4 );
      ehfloat16 t16 = (ehfloat16)( eij.x*t8, eij.y*t8 );
      X += t16*r;
      X1 += t8;
      X2 += t8*r*r;
    }
  }
  ehfloat8 X1dotB = (ehfloat8)(
    X1.even.x*B.lo+X1.odd.x*B.hi,
    X1.even.y*B.lo+X1.odd.y*B.hi,
    X1.even.z*B.lo+X1.odd.z*B.hi,
    X1.even.w*B.lo+X1.odd.w*B.hi
  );
  ehfloat16 X3 = (ehfloat16)(
    X1dotB.even.x*X2.lo+X1dotB.odd.x*X2.hi,
    X1dotB.even.y*X2.lo+X1dotB.odd.y*X2.hi,
    X1dotB.even.z*X2.lo+X1dotB.odd.z*X2.hi,
    X1dotB.even.w*X2.lo+X1dotB.odd.w*X2.hi
  );
  X += X3;

  /*
    X0 X4 X8 Xc
    X1 X5 X9 Xd
    X2 X6 Xa Xe
    X3 X7 Xb Xf

    X0 X4+X8 Xc
    X1 X5+X9 Xd
    X3 X7+Xb Xf
  */
  M[0] = X.s013;
  M[1] = X.s457 + X.s89b;
  M[2] = X.scdf;
  }

  ehfloat Det = dot( M[0], cross(M[1],M[2]) );
  if( fabs(Det) < LAPLACIAN_TENSOR_EPS )
  {
    return (ehfloat4)(1, 0, 0, 1);
  }

  ehfloat3 r0 = cross( M[1], M[2] );
  ehfloat3 r1 = cross( M[2], M[0] );
  ehfloat3 r2 = cross( M[0], M[1] );
  ehfloat3 rhs = (ehfloat3)( -1, 0, -1 );
  ehfloat3 answer = (1.0/Det) * (ehfloat3)( dot(r0,rhs), dot(r1,rhs), dot(r2,rhs) );
  return (ehfloat4)( answer.x, answer.y, answer.y, answer.z );
}

// 非圧力勾配(粘性，重力）計算
kernel void calculate_nonpressure_force(
  constant struct constant_t *c,
  global const int *grid_beginpoint,

  global const ehfloat2 *position,
  global const ehfloat *rho,
  global const ehfloat2 *velocity,
  global const int *flags,
  global ehfloat *omega,
  global ehfloat2 *grad_vx,
  global ehfloat2 *grad_vy,
  global ehfloat2 *nonpressure_force,
  global const ehfloat *V
)
{
  const int id = get_global_id(0);
  if( id >= c->N ){ return; }
  if( flags[id] & EH_PARTICLE_NOFORCE ){ return; }

  int2 index2 = gridindex2_from_p2( c, position[id] );
  int2 mingrid = max( index2-1, 0 );
  int2 maxgrid = min( index2+1, c->gridsize-1 );

  grad_vx[id] = (ehfloat2)(0,0);
  grad_vy[id] = (ehfloat2)(0,0);

/*
  ehfloat4 B = gradient_tensor(c,grid_beginpoint,position,rho,flags,V,position[id],0);
  {
  ehfloat2 gradvx = (ehfloat2)( 0,0 );
  ehfloat2 gradvy = (ehfloat2)( 0,0 );
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int j=begin; j<end; ++j )
    {
      ehfloat2 rij = position[id] - position[j];
      if( dot(rij,rij) > c->H*c->H ){ continue; }

      ehfloat2 kd = kernel_gradient( c->invH, rij );
      ehfloat2 BkdV = (B.xy*kd.x + B.zw*kd.y)*V[j];
      ehfloat2 vji = velocity[j] - velocity[id];

      gradvx += vji.x*BkdV;
      gradvy += vji.y*BkdV;
    }
  }
  grad_vx[id] = gradvx;
  grad_vy[id] = gradvy;
  omega[id] = gradvy.x - gradvx.y;
  }
  */

  ehfloat4 B = (ehfloat4)(0);
  ehfloat4 Bhat = laplacian_tensor(c,grid_beginpoint,position,rho,flags,V,position[id],B,id,0);
  ehfloat4 lapvx = (ehfloat4)(0,0,0,0);
  ehfloat4 lapvy = (ehfloat4)(0,0,0,0);
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int j=begin; j<end; ++j )
    {
      if( j == id ){ continue; }
      ehfloat2 eij = position[id] - position[j];
      if( dot(eij,eij) > c->H*c->H ){ continue; }
      ehfloat2 kdV = kernel_gradient(c->invH,eij)*V[j];
      ehfloat2 vij = velocity[id] - velocity[j];
      ehfloat invlen = 1.0/length(eij);
      eij = normalize(eij);
      ehfloat4 eijkdV = (ehfloat4)( eij*kdV.x, eij*kdV.y );
      lapvx += 2*( vij.x*invlen - dot(eij,grad_vx[id]) ) * eijkdV;
      lapvy += 2*( vij.y*invlen - dot(eij,grad_vy[id]) ) * eijkdV;
    }
  }

  nonpressure_force[id] =
    rho[id]*c->gravity + (ehfloat2)( dot(Bhat,lapvx), dot(Bhat,lapvy) )*c->mu;
}

// 圧力計算
kernel void calculate_pressure(
  constant struct constant_t *c,
  global const ehfloat *rho,
  global const int *flags,
  global const ehfloat2 *position,
  global ehfloat *pressure
  )
{
  int id = get_global_id(0);
  if( id >= c->N ){ return; }
  pressure[id] = c->Cs*c->Cs*c->rho0/c->gamma * pow( rho[id]/c->rho0, c->gamma );
  if( flags[id] & EH_PARTICLE_PRESSURECLAMP )
  {
    pressure[id] = max(pressure[id],c->pressure0);
  }
  
  // 自由表面判定 ( カルマン渦列シミュレーション限定 )
  if( position[id].x > 20-c->H )
  {
    pressure[id] = max(pressure[id],c->pressure0);
  }
}

// 圧力勾配項計算
kernel void calculate_pressure_force(
  constant struct constant_t *c,
  global const int *grid_beginpoint,

  global const ehfloat2 *position,
  global const ehfloat *rho,
  global const ehfloat *pressure,
  global const int *flags,
  global ehfloat2 *pressure_force,
  global const ehfloat *V
)
{
  int id = get_global_id(0);
  if( id >= c->N ){ return; }
  if( flags[id] & EH_PARTICLE_NOFORCE ){ return; }

  int2 index2 = gridindex2_from_p2( c, position[id] );
  int2 mingrid = max( index2-1, 0 );
  int2 maxgrid = min( index2+1, c->gridsize-1 );

  ehfloat4 B = gradient_tensor(c,grid_beginpoint,position,rho,flags,V,position[id],0);
  ehfloat2 force = (ehfloat2)(0,0);
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int j=begin; j<end; ++j )
    {
      ehfloat2 rij = position[id] - position[j];
      if( dot(rij,rij) > c->H*c->H ){ continue; }
      ehfloat2 kd = kernel_gradient(c->invH,rij);
      ehfloat2 BkdV = (B.xy*kd.x+B.zw*kd.y)*V[j];
      force -= (pressure[j]-pressure[id])*BkdV;
    }
  }
  pressure_force[id] = force;
}


// 仮位置，仮速度
kernel void advect_phase1(
  constant struct constant_t *c,
  global const int *flags,
  global const ehfloat2 *svelocity,
  global ehfloat2 *position,
  global ehfloat2 *velocity,
  global const ehfloat *rho,
  global const ehfloat2 *nonpressure_force
)
{
  int id = get_global_id(0);
  if( id >= c->N ){ return; }

  ehfloat2 accel = nonpressure_force[id]/rho[id];
  if( flags[id] & EH_PARTICLE_STATIC ){ accel = (ehfloat2)(0,0); }

  if( flags[id] & EH_PARTICLE_STATICMOVE )
  {
    position[id] += c->dt * svelocity[id];
  }else
  {
    position[id] += c->dt * velocity[id] + 0.5*c->dt*c->dt*accel;
  }
  velocity[id] += c->dt*accel;
}

// 次タイムステップ
// 位置，速度計算
kernel void advect_phase2(
  constant struct constant_t *c,
  global const int *flags,
  global const ehfloat2 *svelocity,
  global ehfloat2 *position,
  global ehfloat2 *velocity,
  global const ehfloat *rho,
  global const ehfloat2 *pressure_force
)
{
  int id = get_global_id(0);
  if( id >= c->N ){ return; }

  ehfloat2 accel = pressure_force[id]/rho[id];
  if( flags[id] & EH_PARTICLE_STATIC ){ accel = (ehfloat2)(0,0); }
  if( (flags[id]&EH_PARTICLE_STATICMOVE) == 0 )
  {
    position[id] += 0.5*c->dt*c->dt*accel;
  }
  velocity[id] += c->dt*accel;
}








struct Nbuffer_t
{
  ehfloat U, D;
  int dummyN;
  int N;
  int left_boundary_flag;
  int right_boundary_flag;
};

// カルマン渦列
// 新しい粒子左から生成
// 右範囲外の粒子削除
kernel void karman_leftright(
  constant struct constant_t *c,

  global ehfloat2 *position,
  global ehfloat2 *velocity,
  global ehfloat2 *svelocity,
  global int *flags,
  global int *colors,
  global struct Nbuffer_t *Nbuffer
)
{
  const int id = get_global_id(0);
  if( id >= c->N ){ return; }

  const ehfloat gap = c->gap;
  if( flags[id]&EH_PARTICLE_LEFT )
  {
    if( position[id].x > 0)
    {
      flags[id] = 0;

      // add new particle
      int newid = atomic_inc( &Nbuffer->N );
      position[newid] = position[id] - (ehfloat2)( (Nbuffer->dummyN)*gap, 0 );
      velocity[newid] = (ehfloat2)( Nbuffer->U, 0 );
      svelocity[newid] = (ehfloat2)( Nbuffer->U, 0 );
      if( position[id].y > 4.5 && position[id].y < 5.5 )
      {
        colors[newid] = 1;
      }else
      {
        colors[newid] = 0;
      }
      flags[newid] = Nbuffer->left_boundary_flag;
    }
  }
  else if( flags[id] == 0 )
  {
    if( position[id].x > 20 )
    {
      position[id].x = 30;
    }
  }
}

ehfloat get_float_at(
  constant struct constant_t *c,
  global const int *grid_beginpoint,
  global const ehfloat2 *position,
  global const ehfloat *rho,
  global const ehfloat *A,
  global const ehfloat *V,
  global const int *flags,
  ehfloat2 point,
  ehfloat H,
  int except_flag
)
{
  int2 index2 = gridindex2_from_p2( c, point );
  int2 mingrid = max( index2-1, 0 );
  int2 maxgrid = min( index2+1, c->gridsize-1 );

  ehfloat Asum = 0;
  ehfloat ksum = 0;
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int j=begin; j<end; ++j )
    {
      if( flags[j] & except_flag ){ continue; }
      ehfloat2 rij = point - position[j];
      if( dot(rij,rij) > H*H ){ continue; }
      ehfloat kV = kernel_function(1.0/H,rij)*V[j];
      ksum += kV;
      Asum += A[j]*kV;
    }
  }
  return Asum/ksum;
}

// 円柱にかかる力
kernel void calculate_cylinder_force(
  constant struct constant_t *c,
  global const int *grid_beginpoint,

  global const ehfloat2 *position,
  global const ehfloat2 *velocity,
  global const int *flags,
  global const ehfloat *rho,
  global const ehfloat *V,
  global const ehfloat *pressure,
  global ehfloat2 *force_buffer,
  global ehfloat2 *out
)
{
  {
    const ehfloat theta = 2.0*EH_PI*get_global_id(0)/get_global_size(0);
    const ehfloat2 normal = (ehfloat2)( cos(theta), sin(theta) );
    const ehfloat2 point = (ehfloat2)(5,5) + normal*0.5;
    const ehfloat ds = 0.5*2.0*EH_PI/get_global_size(0);

    const ehfloat press = get_float_at(c,grid_beginpoint,position,rho,pressure,V,flags,point,c->H,0);

    int2 index2 = gridindex2_from_p2( c, point );
    int2 mingrid = max( index2-1, 0 );
    int2 maxgrid = min( index2+1, c->gridsize-1 );

    ehfloat4 B = gradient_tensor(c,grid_beginpoint,position,rho,flags,V,point,0);
    ehfloat2 gradvx = (ehfloat2)( 0,0 );
    ehfloat2 gradvy = (ehfloat2)( 0,0 );
    for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
    {
      int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
      int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

      for( int j=begin; j<end; ++j )
      {
        ehfloat2 rij = point - position[j];
        if( dot(rij,rij) > c->H*c->H ){ continue; }

        ehfloat2 kd = kernel_gradient( c->invH, rij );
        ehfloat2 BkdV = (B.xy*kd.x + B.zw*kd.y)*V[j];
        ehfloat2 vji = velocity[j];

        gradvx += vji.x*BkdV;
        gradvy += vji.y*BkdV;
      }
    }

    const ehfloat2 dudn = (ehfloat2)( dot(gradvx,normal), dot(gradvy,normal) );
    force_buffer[2*get_global_id(0)] = -press*normal*ds;
    force_buffer[2*get_global_id(0)+1] = c->mu*dudn*ds;
  }
  barrier( CLK_GLOBAL_MEM_FENCE );
  if( get_global_id(0) == 0 )
  {
    ehfloat2 vis = (ehfloat2)(0,0);
    ehfloat2 pre = (ehfloat2)(0,0);
    for( int i=0; i<get_global_size(0); ++i )
    {
      pre += force_buffer[2*i];
      vis += force_buffer[2*i+1];
    }
    out[4] = vis;
    out[5] = pre;
  }
}

ehfloat2 get_vec2_at(
  constant struct constant_t *c,
  global const int *grid_beginpoint,
  global const ehfloat2 *position,
  global const ehfloat *rho,
  global const ehfloat2 *A,
  global const ehfloat *V,
  global const int *flags,
  ehfloat2 point,
  ehfloat H,
  int except_flag
)
{
  int2 index2 = gridindex2_from_p2( c, point );
  int2 mingrid = max( index2-1, 0 );
  int2 maxgrid = min( index2+1, c->gridsize-1 );

  ehfloat2 Asum = (ehfloat2)(0,0);
  ehfloat ksum = 0;
  for( int gridy=mingrid.y; gridy<=maxgrid.y; ++gridy )
  {
    int begin = grid_beginpoint[ gridindex_from_index2(c,(int2)(mingrid.x,gridy)) ];
    int end = grid_beginpoint[ gridindex_from_index2(c,(int2)(maxgrid.x,gridy)) + 1 ];

    for( int j=begin; j<end; ++j )
    {
      if( flags[j] & except_flag ){ continue; }
      ehfloat2 rij = point - position[j];
      if( dot(rij,rij) > H*H ){ continue; }
      ehfloat kV = kernel_function(1.0/H,rij)*V[j];
      ksum += kV;
      Asum += A[j]*kV;
    }
  }
  return Asum/ksum;
}

// 点ｘでの流速
kernel void get_velocity(
  constant struct constant_t *c,
  global const int *grid_beginpoint,
  global const ehfloat2 *position,
  global const ehfloat2 *velocity,
  global const ehfloat *rho,
  global const int *flags,
  global const ehfloat *V,
  global ehfloat2 *out,
  int N
)
{
  const int id = get_global_id(0);
  if( id >= N ){ return; }

  out[id*2+1]
    = get_vec2_at( c, grid_beginpoint, position, rho, velocity, V, flags,
        out[id*2], c->H*0.7, 0 );
}


// ミラー境界
// 円柱壁粒子速度設定; 滑りなし境界条件
kernel void set_cylinder_velocity(
  constant struct constant_t *c,
  global const int *grid_beginpoint,
  global const ehfloat2 *position,
  global ehfloat2 *velocity,
  global const ehfloat *rho,
  global const int *flags,
  global const ehfloat *V
)
{
  const int id = get_global_id(0);
  if( id >= c->N ){ return; }
  if( (flags[id]&EH_PARTICLE_CYLINDER) == 0 ){ return; }

  ehfloat2 normal = position[id] - (ehfloat2)(5,5);
  ehfloat len = length(normal);
  if( len < 0.1 ){ return; }
  normal = normalize(normal);
  ehfloat2 mirror_point = (0.5 + 0.5 - len)*normal + (ehfloat2)(5,5);
  velocity[id] = -get_vec2_at(c,grid_beginpoint,position,rho,velocity,V,flags,mirror_point,c->H, EH_PARTICLE_CYLINDER);
}


// ミラー境界
// 上下端壁粒子の速度設定; 滑り壁境界条件
kernel void set_updown_velocity(
  constant struct constant_t *c,
  global const int *grid_beginpoint,
  global const ehfloat2 *position,
  global ehfloat2 *velocity,
  global const ehfloat *rho,
  global const int *flags,
  global const ehfloat *V
)
{
  const int id = get_global_id(0);
  if( id >= c->N ){ return; }
  if( flags[id] & EH_PARTICLE_UP )
  {
    ehfloat l = position[id].y - 10;
    ehfloat2 mirror_point = (ehfloat2)( position[id].x, 10 - l );
    ehfloat2 mirror_vel = get_vec2_at(c,grid_beginpoint,position,rho,velocity,V,flags,mirror_point,c->H, EH_PARTICLE_UP|EH_PARTICLE_DOWN|EH_PARTICLE_LEFT);
    velocity[id] = (ehfloat2)( mirror_vel.x, -mirror_vel.y );
  }else if( flags[id] & EH_PARTICLE_DOWN )
  {
    ehfloat l = -position[id].y;
    ehfloat2 mirror_point = (ehfloat2)( position[id].x, l );
    ehfloat2 mirror_vel = get_vec2_at(c,grid_beginpoint,position,rho,velocity,V,flags,mirror_point,c->H, EH_PARTICLE_UP|EH_PARTICLE_DOWN|EH_PARTICLE_LEFT);
    velocity[id] = (ehfloat2)( mirror_vel.x, -mirror_vel.y );
  }
}

kernel void get_image(
  constant struct constant_t *c,
  global const int *grid_beginpoint,
  global const ehfloat2 *position,
  global ehfloat2 *velocity,
  global const ehfloat *rho,
  global const int *flags,
  global const ehfloat *V,
  global ehfloat2 *out,
  global ehfloat2 *outpos,
  int W, int H,
  ehfloat H_
)
{
  if( get_global_id(0) >= W ){ return; }
  if( get_global_id(1) >= H ){ return; }

  ehfloat2 point = (ehfloat2)(
    ((ehfloat)(get_global_id(0))+0.5)/W*20.0,
    ((ehfloat)(get_global_id(1))+0.5)/H*10.0
  );
  ehfloat2 vel = get_vec2_at(c,grid_beginpoint,position,rho,velocity,V,flags,point,H_,0);
  out[ get_global_id(1)*W + get_global_id(0) ] = vel;
  outpos[ get_global_id(1)*W + get_global_id(0) ] = point;
}

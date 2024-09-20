#ifndef MACDRP_H
#define MACDRP_H

#define BWD -1
#define FWD 1

#ifndef MAC64

//#define c_1 0.30874
//#define c_2 0.63260
//#define c_3 1.23300
//#define c_4 0.33340
//#define c_5 0.04168

//#define a_0pF (-1.25008)
//#define a_0mF ( 1.24996)
//#define a_0pB (-1.24996)
//#define a_0mB ( 1.25008)
//#define a_1   ( 1.54174)
//#define a_2   (-0.33340)
//#define a_3   ( 0.04168)

#define c_1 0.29164704779069994
#define c_2 0.66674514216519998
#define c_3 1.2501177132452999
#define c_4 0.33341180882999999
#define c_5 0.041686285540599999
//#define c_1 (0.25)
//#define c_2 (5.0/6.0)
//#define c_3 (1.5)
//#define c_4 (0.5)
//#define c_5 (1.0/12.0)

#define a_0pF (-1.250039237746600)
#define a_0mF ( 1.250039237746600)
#define a_0pB (-1.250039237746600)
#define a_0mB ( 1.250039237746600)
#define a_1   ( 1.541764761036000)
#define a_2   (-0.333411808829999)
#define a_3   ( 0.0416862855405999)

#else

#define c_1 0.300000000000000
#define c_2 0.633333333333333
#define c_3 1.200000000000000
#define c_4 0.300000000000000
#define c_5 0.033333333333333

#define a_0pF (-1.233333333333333)
#define a_0mF ( 1.233333333333333)
#define a_0pB (-1.233333333333333)
#define a_0mB ( 1.233333333333333)
#define a_1   ( 1.500000000000000)
#define a_2   (-0.300000000000000)
#define a_3   ( 0.033333333333333)

#endif

#define L(var,idx,stride,FLAG) \
  ((FLAG==1) ? LF(var,idx,stride) : LB(var,idx,stride))

#define LF(var,idx,stride) \
  (-c_1*var[idx-stride  ] \
   -c_2*var[idx         ] \
   +c_3*var[idx+stride  ] \
   -c_4*var[idx+stride*2] \
   +c_5*var[idx+stride*3])

#define LB(var,idx,stride) \
  (-c_5*var[idx-stride*3] \
   +c_4*var[idx-stride*2] \
   -c_3*var[idx-stride  ] \
   +c_2*var[idx         ] \
   +c_1*var[idx+stride  ])

#define LF_old(var,idx,stride) \
  (-0.30874*var[idx-stride  ] \
   -0.6326 *var[idx         ] \
   +1.2330 *var[idx+stride  ] \
   -0.3334 *var[idx+2*stride] \
   +0.04168*var[idx+3*stride])

#define LB_old(var,idx,stride) \
  (-0.04168*var[idx-3*stride] \
   +0.3334 *var[idx-2*stride] \
   -1.2330 *var[idx-stride  ] \
   +0.6326 *var[idx         ] \
   +0.30874*var[idx+stride  ])

#define MacCormack(var, idx, stride, coef) \
  (coef * (-c_1 * var[idx - stride    ] \
           -c_2 * var[idx             ] \
           +c_3 * var[idx + stride    ] \
           -c_4 * var[idx + stride * 2] \
           +c_5 * var[idx + stride * 3]))

#define L22(var,idx,stride,FLAG) \
  ((FLAG==1) ? L22F(var,idx,stride) : L22B(var,idx,stride))

#define L22F(var,idx,stride) \
  (var[idx+stride]-var[idx])

#define L22B(var,idx,stride) \
  (var[idx]-var[idx-stride])

#define MacCormack22(var, idx, stride, coef) \
  (coef * ( var[idx + stride] \
           -var[idx         ]))

// (-7/6 8/6 -1/6)
//#define c24_1 1.16666666666667
//#define c24_2 1.33333333333333
//#define c24_3 0.166666666666667
#define c24_1 (7.0/6.0)
#define c24_2 (4.0/3.0)
#define c24_3 (1.0/6.0)

#define L24(var,idx,stride,FLAG) \
  ((FLAG==1) ? L24F(var,idx,stride) : L24B(var,idx,stride))

#define L24F(var,idx,stride) \
  (-c24_1*var[idx         ] \
   +c24_2*var[idx+stride  ] \
   -c24_3*var[idx+stride*2])

#define L24B(var,idx,stride) \
  ( c24_3*var[idx-stride*2] \
   -c24_2*var[idx-stride  ] \
   +c24_1*var[idx         ])

#define MacCormack24(var, idx, stride, coef) \
  (coef * (-c24_1 * var[idx             ] \
           +c24_2 * var[idx + stride    ] \
           -c24_3 * var[idx + stride * 2]))

#define vec_L22F(var,i) (var[i+1]-var[i])
#define vec_L22B(var,i) (var[i]-var[i-1])

#define vec_L24F(var,i) \
  (-c24_1*var[i  ] \
   +c24_2*var[i+1] \
   -c24_3*var[i+2])

#define vec_L24B(var,i) \
  ( c24_3*var[i-2] \
   -c24_2*var[i-1] \
   +c24_1*var[i  ])

#define vec_LF(var,i) \
  (-c_1*var[i-1] \
   -c_2*var[i  ] \
   +c_3*var[i+1] \
   -c_4*var[i+2] \
   +c_5*var[i+3])

#define vec_LB(var,i) \
  (-c_5*var[i-3] \
   +c_4*var[i-2] \
   -c_3*var[i-1] \
   +c_2*var[i  ] \
   +c_1*var[i+1])

#define vec_L24(var,i,FLAG) \
  ((FLAG==1) ? vec_L24F(var,i) : vec_L24B(var,i))
#define vec_L22(var,i,FLAG) \
  ((FLAG==1) ? vec_L22F(var,i) : vec_L22B(var,i))
#define vec_L(var,i,FLAG) \
  ((FLAG==1) ? vec_LF(var,i) : vec_LB(var,i))

#define compact_a1 1.267949192431123
#define compact_a2 0.267949192431123

// filters
#define f9pd0 ( 0.243527493120)
#define f9pd1 (-0.204788880640)
#define f9pd2 ( 0.120007591680)
#define f9pd3 (-0.045211119360)
#define f9pd4 ( 0.008228661760)

#define f7pd0 ( 0.351061040)
#define f7pd1 (-0.242824317)
#define f7pd2 ( 0.074469480)
#define f7pd3 (-0.007175683)

// non-centered 7-points FD coefs of Berland
#define FD24_1 ( 0.048264094108)
#define FD24_2 (-0.488255830845)
#define FD24_3 (-0.366015590723)
#define FD24_4 ( 1.048005455857)
#define FD24_5 (-0.289325926394)
#define FD24_6 ( 0.050392437692)
#define FD24_7 (-0.003064639693)

#define FD15_1 (-0.212932721951)
#define FD15_2 (-1.060320390770)
#define FD15_3 ( 2.078926116439)
#define FD15_4 (-1.287179452384)
#define FD15_5 ( 0.685176395471)
#define FD15_6 (-0.245320613994)
#define FD15_7 ( 0.041650667189)

#define FD06_1 (-2.225833963270)
#define FD06_2 ( 4.827779580575)
#define FD06_3 (-5.001388453836)
#define FD06_4 ( 3.911103941646)
#define FD06_5 (-2.115267458633)
#define FD06_6 ( 0.718882784412)
#define FD06_7 (-0.115276430895)

#endif

/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifdef __INTELLISENSE__
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
#endif

#include "culbfgsb/lbfgsbcuda.h"

namespace lbfgsbcuda {
namespace cuda {
namespace subsm {

template <int bx, typename real>
__global__ void kernel00(const int nsub, const int* ind, const int head,
                         const int m, const int col, const int iPitch_ws,
                         const int oPitch, real* buf_array_p, const real* wy,
                         const real* ws, const real* d, const real theta) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (j < nsub) {
    int pointr = Modular((head + i % col), m);
    const int k = ind[j];
    if (i >= col) {
      mySum = ws[k * iPitch_ws + pointr] * theta;
    } else {
      mySum = wy[k * iPitch_ws + pointr];
    }
    mySum *= d[j];
  } else {
    mySum = 0;
  }

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = (mySum + sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = (mySum + sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = (mySum + sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = (mySum + sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = mySum + smem[32];
    }
    if (bx > 16) {
      *smem = mySum = mySum + smem[16];
    }
    if (bx > 8) {
      *smem = mySum = mySum + smem[8];
    }
    if (bx > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bx > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bx > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tid == 0) buf_array_p[i * oPitch + blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel01(const int n, const int iPitch, const int oPitch,
                         const real* buf_in, real* buf_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < n)
    mySum = buf_in[j * iPitch + i];
  else
    mySum = 0;

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = (mySum + sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = (mySum + sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = (mySum + sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = (mySum + sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = mySum + smem[32];
    }
    if (bx > 16) {
      *smem = mySum = mySum + smem[16];
    }
    if (bx > 8) {
      *smem = mySum = mySum + smem[8];
    }
    if (bx > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bx > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bx > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tid == 0) {
    buf_out[j * oPitch + blockIdx.x] = mySum;
  }
}

template <typename real>
void prog0(const int n, const int* ind, const int head, const int m,
           const int col, const int iPitch_ws, real* buf_array_p,
           const real* wy, const real* ws, const real* d, real* wv,
           const real theta, const int iPitch_normal,
           const cudaStream_t& stream) {
  int nblock0 = n;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output = (nblock1 == 1) ? wv : buf_array_p;
  int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

  dynamicCall(
      kernel00, mi, real, nblock1, col * 2, stream,
      (n, ind, head, m, col, iPitch_ws, op20, output, wy, ws, d, theta));

  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input = output;

    output = (nblock1 == 1) ? wv : (output + nblock0);

    int op20 = (nblock1 == 1) ? 1 : iPitch_normal;
    dynamicCall(kernel01, mi, real, nblock1, col * 2, stream,
                (nblock0, iPitch_normal, op20, input, output));

    nblock0 = nblock1;
  }
}

template <typename real>
__global__ void kernel1(real* wv) {
  const int i = threadIdx.x;
  wv[i] = -wv[i];
}

template <typename real>
void prog1(real* wn, int col, int iPitch_wn, real* wv,
           cublasHandle_t cublas_handle, const cudaStream_t& stream) {
  int col2 = col * 2;
  debugSync();

  cublasSetStream(cublas_handle, stream);
  cublasRtrsv<real>(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, col2, wn, iPitch_wn, wv, 1);
  debugSync();
  kernel1<real><<<1, col, 0, stream>>>(wv);
  debugSync();
  cublasRtrsv<real>(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    CUBLAS_DIAG_NON_UNIT, col2, wn, iPitch_wn, wv, 1);
  debugSync();
  cublasSetStream(cublas_handle, NULL);
}

template <int bsize, typename real>
__global__ void kernel2(int nsub, const int* ind, const int col, const int head,
                        const int m, const int iPitch, const real* wv,
                        const real* wy, const real* ws, const real inv_theta,
                        real* d) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  const int tidx = threadIdx.x;  // 8
  const int tidy = threadIdx.y;  // 64

  volatile __shared__ real sdata[(512 / bsize)][bsize + 1];

  __shared__ real a[2][bsize + 1];

  real mySum;

  if (tidy == 0 && tidx < col) {
    a[0][tidx] = wv[tidx] * inv_theta;
    a[1][tidx] = wv[col + tidx];
  }
  __syncthreads();
  if (i < nsub && tidx < col) {
    const int pointr = Modular((head + tidx), m);
    const int k = ind[i];
    mySum = wy[k * iPitch + pointr] * a[0][tidx] +
            ws[k * iPitch + pointr] * a[1][tidx];
  } else
    mySum = 0;

  if (bsize > 1) {
    volatile real* smem = sdata[tidy] + tidx;
    *smem = mySum;
    __syncthreads();
    if (bsize > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bsize > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bsize > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tidx == 0 && i < nsub) {
    d[i] = (d[i] + mySum) * inv_theta;
  }
}

template <typename real>
void prog2(const int nsub, const int* ind, const int col, const int head,
           const int m, const int iPitch, const real* wv, const real* wy,
           const real* ws, const real theta, real* d,
           const cudaStream_t& stream) {
  real invtheta = 1.0 / theta;

  if (col > 4) {
    int nblocky = 512 / 8;
    kernel2<8, real>
        <<<dim3(iDivUp(nsub, nblocky)), dim3(8, nblocky), 0, stream>>>(
            nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
  } else if (col > 2) {
    int nblocky = 512 / 4;
    kernel2<4, real>
        <<<dim3(iDivUp(nsub, nblocky)), dim3(4, nblocky), 0, stream>>>(
            nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
  } else if (col > 1) {
    int nblocky = 512 / 2;
    kernel2<2, real>
        <<<dim3(iDivUp(nsub, nblocky)), dim3(2, nblocky), 0, stream>>>(
            nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
  } else if (col == 1) {
    int nblocky = 512 / 1;
    kernel2<1, real>
        <<<dim3(iDivUp(nsub, nblocky)), dim3(1, nblocky), 0, stream>>>(
            nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
  }
}

template <typename real>
__global__ void kernel210(int nsub, const int* ind, const real* d, real* x,
                          const real* l, const real* u, const int* nbd) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nsub) return;

  const int k = ind[i];
  real xk = x[k] + d[i];
  const int nbdk = nbd[k];

  if (nbdk == 1) {
    xk = maxr(l[k], xk);
  } else if (nbdk == 2) {
    xk = maxr(l[k], xk);
    xk = minr(u[k], xk);
  } else if (nbdk == 3) {
    xk = minr(u[k], xk);
  }

  x[k] = xk;
}

template <int bx, typename real>
__global__ void kernel211(const int n, real* buf_n_r, const real* x,
                          const real* xx, const real* gg) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < n) {
    mySum = (x[i] - xx[i]) * gg[i];
  } else {
    mySum = 0;
  }

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = (mySum + sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = (mySum + sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = (mySum + sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = (mySum + sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = mySum + smem[32];
    }
    if (bx > 16) {
      *smem = mySum = mySum + smem[16];
    }
    if (bx > 8) {
      *smem = mySum = mySum + smem[8];
    }
    if (bx > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bx > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bx > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tid == 0) buf_n_r[blockIdx.x] = mySum;
}

template <typename real>
void prog21(int n, int nsub, const int* ind, const real* d, real* x,
            const real* l, const real* u, const int* nbd, const real* xx,
            const real* gg, real* buf_n_r, real* pddp,
            const cudaStream_t& stream) {
  kernel210<real>
      <<<iDivUp(n, 512), 512, 0, stream>>>(nsub, ind, d, x, l, u, nbd);

  int nblock0 = n;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output = (nblock1 == 1) ? pddp : buf_n_r;

  dynamicCall(kernel211, mi, real, nblock1, 1, stream, (n, output, x, xx, gg));

  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input = output;

    output = (nblock1 == 1) ? pddp : (output + nblock0);

    dynamicCall(kernel01, mi, real, nblock1, 1, stream,
                (nblock0, n, 1, input, output));

    nblock0 = nblock1;
  }
}

template <typename real>
__device__ inline void minex(volatile real& a, volatile real& b,
                             volatile int& ia, volatile int& ib) {
  if (a > b) {
    ia = ib, a = b;
  }
}

template <int bx, typename real>
__global__ void kernel30(const int nsub, const int* ind, real* d,
                         const int* nbd, real* t, int* ti, real* x,
                         const real* u, const real* l) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];
  volatile __shared__ int sdatai[bx];

  real mySum = 1.0;

  if (i < nsub) {
    const int k = ind[i];
    const int nbdi = nbd[k];

    if (nbdi != 0) {
      real dk = d[i];
      if (dk < 0 && nbdi <= 2) {
        real temp2 = l[k] - x[k];
        if (temp2 >= 0) {
          mySum = 0;
        } else {
          mySum = minr(static_cast<real>(1.0), temp2 / dk);
        }
      } else if (dk > 0 && nbdi >= 2) {
        real temp2 = u[k] - x[k];
        if (temp2 <= 0) {
          mySum = 0;
        } else {
          mySum = minr(static_cast<real>(1.0), temp2 / dk);
        }
      }
    }
  }

  sdata[tid] = mySum;
  sdatai[tid] = i;
  __syncthreads();

  if (bx > 512) {
    if (tid < 512) {
      minex<real>(sdata[tid], sdata[tid + 512], sdatai[tid], sdatai[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      minex<real>(sdata[tid], sdata[tid + 256], sdatai[tid], sdatai[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      minex<real>(sdata[tid], sdata[tid + 128], sdatai[tid], sdatai[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      minex<real>(sdata[tid], sdata[tid + 64], sdatai[tid], sdatai[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    volatile int* smemi = sdatai + tid;
    if (bx > 32) {
      minex<real>(*smem, smem[32], *smemi, smemi[32]);
    }
    if (bx > 16) {
      minex<real>(*smem, smem[16], *smemi, smemi[16]);
    }
    if (bx > 8) {
      minex<real>(*smem, smem[8], *smemi, smemi[8]);
    }
    if (bx > 4) {
      minex<real>(*smem, smem[4], *smemi, smemi[4]);
    }
    if (bx > 2) {
      minex<real>(*smem, smem[2], *smemi, smemi[2]);
    }
    if (bx > 1) {
      minex<real>(*smem, smem[1], *smemi, smemi[1]);
    }

    if (tid == 0) {
      t[blockIdx.x] = *smem;
      ti[blockIdx.x] = *smemi;

      if (gridDim.x == 1 && *smem < 1) {
        real dk = d[*smemi];
        const int k = ind[*smemi];
        if (dk > 0) {
          x[k] = u[k];
          d[*smemi] = 0;
        } else if (dk < 0) {
          x[k] = l[k];
          d[*smemi] = 0;
        }
      }
    }
  }
}

template <int bx, typename real>
__global__ void kernel31(const int n, const int* ind, const real* buf_in,
                         const int* bufi_in, real* buf_out, int* bufi_out,
                         real* d, real* x, const real* u, const real* l) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];
  volatile __shared__ int sdatai[bx];

  real mySum;
  int mySumi;
  if (i < n) {
    mySum = buf_in[i];
    mySumi = bufi_in[i];
  } else {
    mySum = 1.0;
    mySumi = 0;
  }

  sdata[tid] = mySum;
  sdatai[tid] = mySumi;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      minex<real>(sdata[tid], sdata[tid + 512], sdatai[tid], sdatai[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      minex<real>(sdata[tid], sdata[tid + 256], sdatai[tid], sdatai[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      minex<real>(sdata[tid], sdata[tid + 128], sdatai[tid], sdatai[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      minex<real>(sdata[tid], sdata[tid + 64], sdatai[tid], sdatai[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    volatile int* smemi = sdatai + tid;
    if (bx > 32) {
      minex<real>(*smem, smem[32], *smemi, smemi[32]);
    }
    if (bx > 16) {
      minex<real>(*smem, smem[16], *smemi, smemi[16]);
    }
    if (bx > 8) {
      minex<real>(*smem, smem[8], *smemi, smemi[8]);
    }
    if (bx > 4) {
      minex<real>(*smem, smem[4], *smemi, smemi[4]);
    }
    if (bx > 2) {
      minex<real>(*smem, smem[2], *smemi, smemi[2]);
    }
    if (bx > 1) {
      minex<real>(*smem, smem[1], *smemi, smemi[1]);
    }

    if (tid == 0) {
      buf_out[blockIdx.x] = *smem;
      bufi_out[blockIdx.x] = *smemi;

      if (gridDim.x == 1 && *smem < 1) {
        real dk = d[*smemi];
        const int k = ind[*smemi];
        if (dk > 0) {
          x[k] = u[k];
          d[*smemi] = 0;
        } else if (dk < 0) {
          x[k] = l[k];
          d[*smemi] = 0;
        }
      }
    }
  }
}

template <typename real>
__global__ void kernel32(const int nsub, const int* ind, real* x, const real* d,
                         const real* alpha) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ real salpha[1];

  if (i >= nsub) return;

  const int k = ind[i];

  if (threadIdx.x == 0) {
    *salpha = alpha[0];
  }
  real xi = x[k];
  real di = d[i];

  __syncthreads();

  x[k] = salpha[0] * di + xi;
}

template <typename real>
void prog3(const int nsub, const int* ind, real* d, const int* nbd,
           real* buf_s_r, int* bufi_s_r, real* x, const real* u, const real* l,
           const cudaStream_t& stream) {
  int nblock0 = nsub;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output_r = buf_s_r;
  int* output_i = bufi_s_r;

  dynamicCall(kernel30, mi, real, nblock1, 1, stream,
              (nsub, ind, d, nbd, output_r, output_i, x, u, l));

  debugSync();
  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input_r = output_r;
    int* input_i = output_i;

    output_r = output_r + nblock0;
    output_i = output_i + nblock0;

    dynamicCall(
        kernel31, mi, real, nblock1, 1, stream,
        (nblock0, ind, input_r, input_i, output_r, output_i, d, x, u, l));

    nblock0 = nblock1;
  }

  kernel32<real><<<dim3(iDivUp(nsub, 512)), dim3(512), 0, stream>>>(
      nsub, ind, x, d, output_r);
}

#define INST_HELPER(real)                                                    \
  template void prog0<real>(const int, const int*, const int, const int,     \
                            const int, const int, real*, const real*,        \
                            const real*, const real*, real*, const real,     \
                            const int, const cudaStream_t&);                 \
  template void prog1<real>(real*, int, int, real*, cublasHandle_t,          \
                            const cudaStream_t&);                            \
  template void prog2<real>(const int, const int*, const int, const int,     \
                            const int, const int, const real*, const real*,  \
                            const real*, const real, real*,                  \
                            const cudaStream_t&);                            \
  template void prog21<real>(int, int, const int*, const real*, real*,       \
                             const real*, const real*, const int*,           \
                             const real*, const real*, real*, real*,         \
                             const cudaStream_t&);                           \
  template void prog3(const int, const int*, real*, const int*, real*, int*, \
                      real*, const real*, const real*, const cudaStream_t&); 

INST_HELPER(double);
INST_HELPER(float);
};  // namespace subsm
}  // namespace cuda
};  // namespace lbfgsbcuda
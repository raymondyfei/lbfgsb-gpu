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
namespace formk {

template <typename real>
__global__ void kernel0(real* wn1, const int m, const int iPitch) {
  const int i = threadIdx.x;
  const int j = threadIdx.y;

  __shared__ real sdata[16][16];

  if (j >= m * 2 || i >= m * 2 || i > j) {
    sdata[j][i] = 0;
  } else {
    sdata[j][i] = wn1[j * iPitch + i];
  }
  __syncthreads();

  if (j < m * 2 - 1 && i < m * 2 - 1 && i <= j && j != m - 1 && i != m - 1)
    wn1[j * iPitch + i] = sdata[j + 1][i + 1];
}

template <int bx, typename real>
__global__ void kernel10(const int n, const int nsub, const int ipntr,
                         real* output, const real* wy, const int* ind,
                         const int head, const int m, const int iPitch_ws,
                         const int oPitch) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < nsub) {
    const int i1 = ind[i];
    const int jpntr = Modular((head + j), m);
    mySum = wy[i1 * iPitch_ws + ipntr] * wy[i1 * iPitch_ws + jpntr];
  } else
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

  if (tid == 0) output[j * oPitch + blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel101(const int n, const int nsub, const int ipntr,
                          real* output, const real* ws, const int* ind,
                          const int head, const int m, const int iPitch_ws,
                          const int oPitch) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  const int k = nsub + i;

  if (k < n) {
    const int k1 = ind[k];
    const int jpntr = Modular((head + j), m);
    mySum = ws[k1 * iPitch_ws + ipntr] * ws[k1 * iPitch_ws + jpntr];
  } else
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

  if (tid == 0) output[j * oPitch + blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel102(const int n, const int nsub, const int ipntr,
                          real* output, const real* ws, const real* wy,
                          const int* ind, const int head, const int m,
                          const int iPitch_ws, const int oPitch) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  const int k = nsub + i;

  if (k < n) {
    const int k1 = ind[k];
    const int jpntr = Modular((head + j), m);
    mySum = ws[k1 * iPitch_ws + ipntr] * wy[k1 * iPitch_ws + jpntr];
  } else
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

  if (tid == 0) output[j * oPitch + blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel11(const int n, const int iPitch, const int oPitch,
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

template <int bx, typename real>
__global__ void kernel311(const int n, const int iPitch, const int oPitch,
                          const int2* pcoord, const real* buf_in,
                          real* buf_out) {
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
    if (gridDim.x == 1) {
      const int2 coord = pcoord[i];
      int jy = coord.x;
      int iy = coord.y;

      buf_out[iy * oPitch + jy] += mySum;
    } else {
      buf_out[j * oPitch + blockIdx.x] = mySum;
    }
  }
}

template <int bx, typename real>
__global__ void kernel321(const int n, const int iPitch, const int oPitch,
                          const real* buf_in, real* buf_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y;
  const int iy = blockIdx.z;
  const int j = iy * gridDim.y + jy;
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
    if (gridDim.x == 1) {
      buf_out[iy * oPitch + jy] += mySum;
    } else {
      buf_out[j * oPitch + blockIdx.x] = mySum;
    }
  }
}

template <int bx, typename real>
__global__ void kernel30(const int* ind, const int jpntr, const int head,
                         const int m, const int n, const int nsub,
                         const int iPitch_ws, const real* ws, const real* wy,
                         real* output, const int oPitch) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (k < nsub) {
    const int i1 = ind[k];
    const int ipntr = Modular((head + i), m);
    mySum = ws[i1 * iPitch_ws + ipntr] * wy[i1 * iPitch_ws + jpntr];
  } else
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

  if (tid == 0) output[i * oPitch + blockIdx.x] = mySum;
}

template <typename real>
__global__ void kernel50(real* wn) { wn[1] = wn[1] / wn[0]; }

template <typename real>
__global__ void kernel5(int col, int iPitch_wn, real* wn) {
  const int iis = blockIdx.x + col;
  const int js = threadIdx.y + col;
  const int i = threadIdx.x;

  volatile __shared__ real sdata[64];

  real mySum = 0;
  if (blockIdx.y < col && blockIdx.x < col && js >= iis && i < col) {
    mySum = wn[i * iPitch_wn + iis] * wn[i * iPitch_wn + js];
  }

  volatile real* smem = sdata + (threadIdx.y * blockDim.x + i);
  *smem = mySum;
  __syncthreads();

  if (i < 4) {
    *smem = mySum = mySum + smem[4];
    *smem = mySum = mySum + smem[2];
    *smem = mySum = mySum + smem[1];
  }

  if (i == 0) wn[iis * iPitch_wn + js] += mySum;
}

template <typename real>
void prog0(real* wn1, int m, int iPitch_wn, const cudaStream_t* streamPool) {
  kernel0<real><<<1, dim3(16, 16)>>>(wn1, m, iPitch_wn);
}

template <typename real>
void prog1(const int n, const int nsub, const int ipntr, const int* ind,
           real* wn1, real* buf_array_p, const real* ws, const real* wy,
           const int head, const int m, const int col, const int iPitch_ws,
           const int iPitch_wn, const int iPitch_normal,
           const cudaStream_t* streamPool) {
  debugSync();

  {
    int nblock0 = nsub;
    int mi = log2Up(nblock0);
    int nblock1 = iDivUp2(nblock0, mi);

    real* output = (nblock1 == 1) ? (wn1 + (col - 1) * iPitch_wn) : buf_array_p;
    int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

    dynamicCall(kernel10, mi, real, nblock1, col, streamPool[0],
                (n, nsub, ipntr, output, wy, ind, head, m, iPitch_ws, op20));

    nblock0 = nblock1;
    while (nblock0 > 1) {
      nblock1 = iDivUp2(nblock0, mi);

      real* input = output;

      output =
          (nblock1 == 1) ? (wn1 + (col - 1) * iPitch_wn) : (output + nblock0);

      int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

      dynamicCall(kernel11, mi, real, nblock1, col, streamPool[0],
                  (nblock0, iPitch_normal, op20, input, output));

      nblock0 = nblock1;
    }
  }
  debugSync();
  if (n > nsub) {
    {
      int nblock0 = n - nsub;
      int mi = log2Up(nblock0);
      int nblock1 = iDivUp2(nblock0, mi);

      real* output = (nblock1 == 1)
                         ? (wn1 + (col - 1) * iPitch_wn + m * (iPitch_wn + 1))
                         : buf_array_p;
      int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

      dynamicCall(kernel101, mi, real, nblock1, col, streamPool[0],
                  (n, nsub, ipntr, output, ws, ind, head, m, iPitch_ws, op20));

      /*
                              kernel10<<<dim3(nblock1, col), dim3(512)>>>
                                      (nsub, ipntr, output, wy, head, m,
         iPitch_ws, op20);*/

      nblock0 = nblock1;
      while (nblock0 > 1) {
        nblock1 = iDivUp2(nblock0, mi);

        real* input = output;

        output = (nblock1 == 1)
                     ? (wn1 + (col - 1) * iPitch_wn + m * (iPitch_wn + 1))
                     : (output + nblock0);

        int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

        dynamicCall(kernel11, mi, real, nblock1, col, streamPool[0],
                    (nblock0, iPitch_normal, op20, input, output));

        nblock0 = nblock1;
      }
    }

    debugSync();

    {
      int nblock0 = n - nsub;
      int mi = log2Up(nblock0);
      int nblock1 = iDivUp2(nblock0, mi);

      real* output =
          (nblock1 == 1) ? (wn1 + (m + col - 1) * iPitch_wn) : buf_array_p;
      int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

      dynamicCall(
          kernel102, mi, real, nblock1, col, streamPool[0],
          (n, nsub, ipntr, output, ws, wy, ind, head, m, iPitch_ws, op20));

      /*
                              kernel10<<<dim3(nblock1, col), dim3(512)>>>
                                      (nsub, ipntr, output, wy, head, m,
         iPitch_ws, op20);*/

      nblock0 = nblock1;
      while (nblock0 > 1) {
        nblock1 = iDivUp2(nblock0, mi);

        real* input = output;

        output = (nblock1 == 1) ? (wn1 + (m + col - 1) * iPitch_wn)
                                : (output + nblock0);

        int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

        dynamicCall(kernel11, mi, real, nblock1, col, streamPool[0],
                    (nblock0, iPitch_normal, op20, input, output));

        nblock0 = nblock1;
      }
    }

    debugSync();
  }
}

template <typename real>
void prog2(real* wn1, const int col, const int m, const int iPitch_wn,
           const cudaStream_t* streamPool) {
  int offset = (col + m - 1) * iPitch_wn;

  cudaMemsetAsync(wn1 + offset + m, 0, col * sizeof(real), streamPool[1]);
  cudaMemsetAsync(wn1 + offset, 0, col * sizeof(real), streamPool[1]);
}

template <typename real>
void prog3(const int* ind, const int jpntr, const int head, const int m,
           const int col, const int n, const int nsub, const int iPitch_ws,
           const int iPitch_wn, const int jy, const real* ws, const real* wy,
           real* buf_array_p, real* wn1, const int iPitch_normal,
           const cudaStream_t* streamPool) {
  int nblock0 = nsub;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output = (nblock1 == 1) ? (wn1 + m * iPitch_wn + jy) : buf_array_p;
  int op20 = (nblock1 == 1) ? iPitch_wn : iPitch_normal;

  dynamicCall(kernel30, mi, real, nblock1, col, streamPool[2],
              (ind, jpntr, head, m, n, nsub, iPitch_ws, ws, wy, output, op20));

  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input = output;

    output = (nblock1 == 1) ? (wn1 + m * iPitch_wn + jy) : (output + nblock0);

    int op20 = (nblock1 == 1) ? iPitch_wn : iPitch_normal;
    dynamicCall(kernel11, mi, real, nblock1, col, streamPool[2],
                (nblock0, iPitch_normal, op20, input, output));

    nblock0 = nblock1;
  }
}

template <int bx, typename real>
__global__ void kernel310(const int* indx2, const int head, const int m,
                          const int n, const int nenter, const int ileave,
                          const int iPitch_ws, const int2* pcoord,
                          const real* wy, const real scal, real* output,
                          const int oPitch) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  int iy = 0;
  int jy = 0;
  if (k < n) {
    const int2 coord = pcoord[i];
    jy = coord.x;
    iy = coord.y;

    const int ipntr = Modular((head + iy), m);
    const int jpntr = Modular((head + jy), m);

    real temp3 = 0;

    if (k <= nenter) {
      temp3 = 1.0;
    } else if (k >= ileave) {
      temp3 = -1.0;
    }

    if (temp3 != 0) {
      const int k1 = indx2[k];
      mySum = wy[k1 * iPitch_ws + ipntr] * wy[k1 * iPitch_ws + jpntr] * temp3 *
              scal;
    } else {
      mySum = 0;
    }
  } else
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
    if (gridDim.x == 1)
      output[iy * oPitch + jy] += mySum;
    else
      output[i * oPitch + blockIdx.x] = mySum;
  }
}

template <int bx, typename real>
__global__ void kernel320(const int* indx2, const int head, const int m,
                          const int n, const int nenter, const int ileave,
                          const int iPitch_ws, const real* ws, const real* wy,
                          real* output, const int oPitch) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int jy = blockIdx.y;
  const int iy = blockIdx.z;
  const int i = iy * gridDim.y + jy;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (k < n) {
    const int ipntr = Modular((head + iy), m);
    const int jpntr = Modular((head + jy), m);

    real temp3 = 0;

    if (k <= nenter) {
      temp3 = 1.0;
    } else if (k >= ileave) {
      temp3 = -1.0;
    }

    if (temp3 != 0) {
      const int k1 = indx2[k];
      if (iy <= jy) {
        temp3 = -temp3;
      }
      mySum = ws[k1 * iPitch_ws + ipntr] * wy[k1 * iPitch_ws + jpntr] * temp3;
    } else {
      mySum = 0;
    }
  } else
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
    if (gridDim.x == 1)
      output[iy * oPitch + jy] += mySum;
    else
      output[i * oPitch + blockIdx.x] = mySum;
  }
}

template <typename real>
void prog31(const int* indx2, const int head, const int m, const int upcl,
            const int col, const int nenter, const int ileave, const int n,
            const int iPitch_ws, const int iPitch_wn, const real* wy,
            real* buf_array_sup, real* wn1, const real scal,
            const int iPitch_super, const cudaStream_t* streamPool) {
  int nblock0 = n;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  int numline = upcl * (upcl + 1) / 2;
  int2* pCoord = new int2[numline];
  int q = 0;
  for (int iy = 0; iy < upcl; iy++) {
    for (int jy = 0; jy <= iy; jy++) {
      pCoord[q].x = jy;
      pCoord[q].y = iy;
      q++;
    }
  }
  int2* pCoord_dev = NULL;
  cutilSafeCall(cudaMalloc(&pCoord_dev, numline * sizeof(int2)));

  cutilSafeCall(cudaMemcpy(pCoord_dev, pCoord, numline * sizeof(int2),
                           cudaMemcpyHostToDevice));

  real* output = (nblock1 == 1) ? wn1 : buf_array_sup;
  int op20 = (nblock1 == 1) ? iPitch_wn : iPitch_super;

  dynamicCall(kernel310, mi, real, nblock1, numline, streamPool[2],
              (indx2, head, m, n, nenter, ileave, iPitch_ws, pCoord_dev, wy,
               scal, output, op20));

  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input = output;

    output = (nblock1 == 1) ? wn1 : (output + nblock0);

    int op20 = (nblock1 == 1) ? iPitch_wn : iPitch_super;
    dynamicCall(kernel311, mi, real, nblock1, numline, streamPool[2],
                (nblock0, iPitch_super, op20, pCoord_dev, input, output));

    nblock0 = nblock1;
  }

  cudaFree(pCoord_dev);
  delete[] pCoord;
}

template <typename real>
void prog32(const int* indx2, const int head, const int m, const int upcl,
            const int nenter, const int ileave, const int n,
            const int iPitch_ws, const int iPitch_wn, const real* wy,
            const real* ws, real* buf_array_sup, real* wn1,
            const int iPitch_super, const cudaStream_t* streamPool) {
  int nblock0 = n;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output = (nblock1 == 1) ? (wn1 + m * iPitch_wn) : buf_array_sup;
  int op20 = (nblock1 == 1) ? iPitch_wn : iPitch_super;

  dynamicCall3D(
      kernel320, mi, real, nblock1, upcl, upcl, streamPool[2],
      (indx2, head, m, n, nenter, ileave, iPitch_ws, ws, wy, output, op20));

  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input = output;

    output = (nblock1 == 1) ? (wn1 + m * iPitch_wn) : (output + nblock0);

    int op20 = (nblock1 == 1) ? iPitch_wn : iPitch_super;
    dynamicCall3D(kernel321, mi, real, nblock1, upcl, upcl, streamPool[2],
                  (nblock0, iPitch_super, op20, input, output));

    nblock0 = nblock1;
  }
}

template <typename real>
__global__ void kernel4(const int col, const int iPitch_wn, const int iPitch_ws,
                        const int m, const real* wn1, const real theta,
                        const real* sy, real* wn) {
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;
  const int jy = blockIdx.x * blockDim.x + threadIdx.x;

  if (iy >= col * 2 || jy > iy) return;

  if (jy < col && jy == iy) {
    wn[iy * iPitch_wn + iy] =
        wn1[iy * iPitch_wn + iy] / theta + sy[iy * iPitch_ws + iy];
  } else if (jy < col - 1 && iy < col && iy > 0) {
    wn[jy * iPitch_wn + iy] = wn1[iy * iPitch_wn + jy] / theta;
  } else if (jy >= col && iy >= col) {
    wn[jy * iPitch_wn + iy] =
        wn1[(m - col + iy) * iPitch_wn + (m - col + jy)] * theta;
  } else if (jy < col - 1 && jy + col < iy && iy >= col + 1) {
    wn[jy * iPitch_wn + iy] = -wn1[(m - col + iy) * iPitch_wn + jy];
  } else if (jy < col && jy + col >= iy && iy >= col) {
    wn[jy * iPitch_wn + iy] = wn1[(m - col + iy) * iPitch_wn + jy];
  }
}

template <typename real>
void prog4(const int col, const int iPitch_wn, const int iPitch_ws, const int m,
           const real* wn1, const real theta, const real* sy, real* wn,
           const cudaStream_t* streamPool) {
  int nblock = iDivUp(col * 2, 8);
  kernel4<real><<<dim3(nblock, nblock), dim3(8, 8), 0, streamPool[2]>>>(
      col, iPitch_wn, iPitch_ws, m, wn1, theta, sy, wn);
}

template <typename real>
void prog5(const int col, const int iPitch_wn, real* wn,
           cublasHandle_t cublas_handle, const cudaStream_t* streamPool) {
  real alpha = 1;
  if (col == 1) {
    kernel50<<<1, 1, 0, streamPool[2]>>>(wn);
  } else {
    cublasSetStream(cublas_handle, streamPool[2]);
    cublasRtrsm<real>(cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                      CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, col, col, &alpha, wn,
                      iPitch_wn, wn + col, iPitch_wn);
    cublasSetStream(cublas_handle, NULL);
    debugSync();
  }
  kernel5<real><<<dim3(col), dim3(8, col), 0, streamPool[2]>>>(col, iPitch_wn, wn);
}

#define INST_HELPER(real)                                                      \
  template void prog0<real>(real*, int, int, const cudaStream_t*);             \
  template void prog1<real>(const int, const int, const int, const int*,       \
                            real*, real*, const real*, const real*, const int, \
                            const int, const int, const int, const int,        \
                            const int, const cudaStream_t*);                   \
  template void prog2<real>(real*, const int, const int, const int,            \
                            const cudaStream_t*);                              \
  template void prog3<real>(const int*, const int, const int, const int,       \
                            const int, const int, const int, const int,        \
                            const int, const int, const real*, const real*,    \
                            real*, real*, const int, const cudaStream_t*);     \
  template void prog31<real>(const int*, const int, const int, const int,      \
                             const int, const int, const int, const int,       \
                             const int, const int, const real*, real*, real*,  \
                             const real, const int, const cudaStream_t*);      \
  template void prog32<real>(const int*, const int, const int, const int,      \
                             const int, const int, const int, const int,       \
                             const int, const real*, const real*, real*,       \
                             real*, const int, const cudaStream_t*);           \
  template void prog4<real>(const int, const int, const int, const int,        \
                            const real*, const real, const real*, real*,       \
                            const cudaStream_t*);                              \
  template void prog5<real>(const int, const int, real*, cublasHandle_t,       \
                            const cudaStream_t*);

INST_HELPER(double);
INST_HELPER(float);

};  // namespace formk
};  // namespace cuda
};  // namespace lbfgsbcuda
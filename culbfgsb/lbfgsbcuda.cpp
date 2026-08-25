/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "lbfgsbcuda.h"

#include <cublas_v2.h>

#include <iostream>

#include "cutil_inline.h"

namespace lbfgsbcuda {
namespace cuda {
template <typename real>
bool checkAvailabilty(const LBFGSB_CUDA_STATE<real>& state) {
  if (!state.m_cublas_handle) {
    std::cerr << "ERROR: CUBLAS HANDLE UNINITIALIZED!" << std::endl;
    std::cerr << "Please check the installation of NVIDIA CUDA" << std::endl;
    return false;
  }

  if (!state.m_funcgrad_callback) {
    std::cerr << "ERROR: CALLBACK IS NULL POINTER!" << std::endl;
    std::cerr
        << "Please call init() to set the callback pointer before minimization"
        << std::endl;
    return false;
  }

  return true;
}

template <typename real>
void lbfgsbminimize(const int& n, const LBFGSB_CUDA_STATE<real>& state,
                    const LBFGSB_CUDA_OPTION<real>& option, real* x,
                    const int* nbd, const real* l, const real* u,
                    LBFGSB_CUDA_SUMMARY<real>& summary) {
  summary.info = 0;
  summary.num_iteration = 0;
  summary.residual_f = summary.residual_g = summary.residual_x =
      option.machine_maximum;

  if (!checkAvailabilty<real>(state)) {
    summary.info = -1;
    return;
  }

  real f;
  real* g;
  real* xold;
  real* xdiff;
  real* z;
  real* xp;
  real* zb;
  real* r;
  real* d;
  real* t;
  real* wa;
  real* buf_n_r;
  real* workvec;
  real* workvec2;

  real* workmat;
  real* ws;
  real* wy;
  real* sy;
  real* ss;
  real* yy;
  real* wt;
  real* wn;
  real* snd;
  real* buf_array_p;
  real* buf_array_p1;
  real* buf_array_super;

  int* bufi_n_r;
  int* iwhere;
  int* index;
  int* iorder;

  int* temp_ind1;
  int* temp_ind2;
  int* temp_ind3;
  int* temp_ind4;

  int csave = 0;
  int task = 0;
  bool updatd = false;
  bool wrk = false;
  int iback = 0;
  int head = 0;
  int col = 0;
  int iter = 0;
  int itail = 0;
  int iupdat = 0;
  int nint = 0;
  int nfgv = 0;
  int internalinfo = 0;
  int ifun = 0;
  int nfree = n;
  int nenter = n;
  int ileave = 0;

  real theta = 1.0;
  real fold = option.machine_maximum;
  real dr = 1.0;
  real rr = 0.0;
  real dnrm = 0.0;
  real xstep = 0.0;
  real sbgnrm = 0.0;
  real ddum = 0.0;
  real dtd = 0.0;
  real gd = 0.0;
  real gdold = 0.0;
  real stp = 0.0;
  real stpmx = 1.0;
  real tf = 0.0;

  int m = std::min(option.hessian_approximate_dimension, 8);

  memAlloc<real>(&workvec, m);
  memAlloc<real>(&workvec2, 2 * m);
  memAlloc<real>(&g, n);
  memAlloc<real>(&xold, n);
  memAlloc<real>(&xdiff, n);
  memAlloc<real>(&z, n);
  memAlloc<real>(&xp, n);
  memAlloc<real>(&zb, n);
  memAlloc<real>(&r, n);
  memAlloc<real>(&d, n);
  memAlloc<real>(&t, n);
  memAlloc<real>(&wa, 8 * m);

  const int superpitch = iDivUp(n, ((1 << log2Up(n)) - 1) / 2);
  const int normalpitch = superpitch;
  memAlloc<real>(&buf_n_r, superpitch);
  memAlloc<real>(&buf_array_p, m * normalpitch * 2);
#ifdef USE_STREAM
  memAlloc<real>(&buf_array_p1, m * normalpitch * 2);
#else
  buf_array_p1 = buf_array_p;
#endif
  memAlloc<real>(&buf_array_super, m * m * superpitch);

  size_t pitch0 = m;
  memAllocPitch<real>(&ws, m, n, &pitch0);
  memAllocPitch<real>(&wy, m, n, NULL);
  memAllocPitch<real>(&sy, m, m, NULL);
  memAllocPitch<real>(&ss, m, m, NULL);
  memAllocPitch<real>(&yy, m, m, NULL);
  memAllocPitch<real>(&wt, m, m, NULL);
  memAllocPitch<real>(&workmat, m, m, NULL);

  size_t pitch1 = m * 2;
  memAllocPitch<real>(&wn, m * 2, m * 2, &pitch1);
  memAllocPitch<real>(&snd, m * 2, m * 2, NULL);

  memAlloc<int>(&bufi_n_r, superpitch);
  memAlloc<int>(&iwhere, n);
  memAlloc<int>(&index, n);
  memAlloc<int>(&iorder, n);

  memAlloc<int>(&temp_ind1, n);
  memAlloc<int>(&temp_ind2, n);
  memAlloc<int>(&temp_ind3, n);
  memAlloc<int>(&temp_ind4, n);

  real* sbgnrm_h;
  real* sbgnrm_d;

  real* dsave13;
  memAllocHost<real>(&sbgnrm_h, &sbgnrm_d, sizeof(real));
  memAllocHost<real>(&dsave13, NULL, 16 * sizeof(real));

  int* isave2 = (int*)(dsave13 + 13);

  real epsx2 = option.eps_x * option.eps_x;

  cudaStream_t streamPool[16] = {NULL};

  const static int MAX_STREAM = 10;

#ifdef USE_STREAM
  for (int i = 0; i < MAX_STREAM; i++)
    cutilSafeCall(cudaStreamCreate(streamPool + i));
#endif

  debugSync();
  lbfgsbactive<real>(n, l, u, nbd, x, iwhere);
  debugSync();
  memCopyAsync(xold, x, n * sizeof(real), cudaMemcpyDeviceToDevice);
  memCopyAsync(xp, x, n * sizeof(real), cudaMemcpyDeviceToDevice);
  if (state.m_funcgrad_callback)
    state.m_funcgrad_callback(x, f, g, NULL, summary);

  nfgv = 1;
  lbfgsbprojgr<real>(n, l, u, nbd, x, g, buf_n_r, sbgnrm_h, sbgnrm_d,
                     option.machine_maximum, NULL);
  cutilSafeCall(cudaDeviceSynchronize());
  sbgnrm = *sbgnrm_h;
  summary.residual_g = sbgnrm;
  if (sbgnrm <= option.eps_g) {
    summary.info = 4;
    return;
  }
  while (true) {
    // Streaming Start
    /*		cutilSafeCall(cudaDeviceSynchronize());*/
    lbfgsbcauchy<real>(n, x, l, u, nbd, g, t, z, zb, m, wy, ws, sy, pitch0, wt,
                       theta, col, head, wa, wa + 2 * m, wa + 6 * m, nint,
                       sbgnrm, buf_n_r, buf_array_p1, iwhere, normalpitch,
                       option.machine_maximum, state.m_cublas_handle,
                       streamPool + 3, internalinfo);
    if (internalinfo != 0) {
      internalinfo = 0;
      col = 0;
      head = 0;
      theta = 1;
      iupdat = 0;
      updatd = false;
      continue;
    }

    freev::prog0(n, nfree, index, nenter, ileave, iorder, iwhere, wrk, updatd,
                 true, iter, temp_ind1, temp_ind2, temp_ind3, temp_ind4);
    // printf("nf/ne/il: %d/%d/%d\n", nfree, nenter, ileave);
    /*		cudaDeviceSynchronize();*/

    if (col != 0 && nfree != 0) {
      if (wrk) {
        lbfgsbformk<real>(n, nfree, index, nenter, ileave, iorder, iupdat,
                          updatd, wn, snd, m, ws, wy, sy, theta, col, head,
                          internalinfo, workvec, workmat, buf_array_p,
                          buf_array_super, pitch1, pitch0, superpitch,
                          normalpitch, option.machine_epsilon,
                          state.m_cublas_handle, streamPool);
        if (internalinfo != 0) {
          internalinfo = 0;
          col = 0;
          head = 0;
          theta = 1;
          iupdat = 0;
          updatd = false;
          continue;
        }
      }

      lbfgsbcmprlb<real>(n, m, x, g, ws, wy, sy, wt, zb, r, wa, index, theta,
                         col, head, nfree, true, internalinfo, workvec,
                         workvec2, pitch0, state.m_cublas_handle,
                         streamPool[0]);
      if (internalinfo != 0) {
        internalinfo = 0;
        col = 0;
        head = 0;
        theta = 1;
        iupdat = 0;
        updatd = false;
        continue;
      }

      lbfgsbsubsm<real>(n, m, nfree, index, l, u, nbd, z, r, xp, ws, wy, theta,
                        x, g, col, head, wa, wn, internalinfo, pitch1, pitch0,
                        buf_array_p, buf_n_r, bufi_n_r, normalpitch,
                        state.m_cublas_handle, streamPool[0]);

      if (internalinfo != 0) {
        internalinfo = 0;
        col = 0;
        head = 0;
        theta = 1;
        iupdat = 0;
        updatd = false;
        continue;
      }
    }
    minimize::vsub_v(n, z, x, d);
    debugSync();
    task = 0;
    while (true) {
      debugSync();
      lbfgsblnsrlb<real>(n, l, u, nbd, x, f, fold, gd, gdold, g, d, r, t, z,
                         stp, dnrm, dtd, xstep, stpmx, option.step_scaling,
                         iter, ifun, iback, nfgv, internalinfo, task, csave,
                         isave2, dsave13, buf_n_r, state.m_cublas_handle,
                         streamPool);
      if (internalinfo != 0 || iback >= 20 || task != 1) {
        break;
      }
      if (state.m_funcgrad_callback)
        state.m_funcgrad_callback(x, f, g, streamPool[1], summary);
      debugSync();
    }
    iter = iter + 1;
    summary.num_iteration = iter;
    if (state.m_after_iteration_callback)
      state.m_after_iteration_callback(x, f, g, NULL, summary);

    lbfgsbprojgr<real>(n, l, u, nbd, x, g, buf_n_r, sbgnrm_h, sbgnrm_d,
                       option.machine_maximum, streamPool[1]);

    minimize::vdiffxchg_v<real>(n, xdiff, xold, x, streamPool[2]);
    debugSync();
    minimize::vdot_vv<real>(n, xdiff, xdiff, tf, state.m_cublas_handle,
                            streamPool[2]);

    minimize::vsub_v<real>(n, g, r, r, streamPool[3]);
    minimize::vdot_vv<real>(n, r, r, rr, state.m_cublas_handle, streamPool[3]);

    ddum = fmaxf(fabs(fold), fmaxf(fabs(f), real(1)));

    summary.residual_f = (fold - f) / ddum;
    if (fold - f <= option.eps_f * ddum) {
      summary.info = 1;
      break;
    }
    if (iter > option.max_iteration && option.max_iteration > 0) {
      summary.info = 5;
      break;
    }
    if (state.m_customized_stopping_callback &&
        state.m_customized_stopping_callback(x, f, g, NULL, summary)) {
      summary.info = 0;
      break;
    }
    cutilSafeCall(cudaDeviceSynchronize());
    if (stp == 1) {
      dr = gd - gdold;
      ddum = -gdold;
    } else {
      dr = (gd - gdold) * stp;
      minimize::vmul_v<real>(n, d, stp, streamPool[1]);
      ddum = -gdold * stp;
    }
    summary.residual_x = sqrt(tf);
    if (tf <= epsx2) {
      summary.info = 2;
      break;
    }
    sbgnrm = *sbgnrm_h;
    summary.residual_g = sbgnrm;
    if (sbgnrm <= option.eps_g) {
      summary.info = 4;
      break;
    }
    if (dr <= option.machine_epsilon * ddum) {
      updatd = false;
    } else {
      updatd = true;
      // 			if(iupdat >= m)
      // 				printf(" ");
      iupdat++;
      /*			printf("iupd: %d\n", iupdat);*/

      lbfgsbmatupd<real>(n, m, ws, wy, sy, ss, d, r, itail, iupdat, col, head,
                         theta, rr, dr, stp, dtd, pitch0, buf_array_p,
                         normalpitch, streamPool);
      lbfgsbformt<real>(m, wt, sy, ss, col, theta, internalinfo, pitch0,
                        option.machine_epsilon, streamPool);
    }

    if (internalinfo != 0) {
      memCopyAsync(x, t, n * sizeof(real), cudaMemcpyDeviceToDevice,
                   streamPool[1]);
      memCopyAsync(g, r, n * sizeof(real), cudaMemcpyDeviceToDevice,
                   streamPool[1]);
      f = fold;

      if (col == 0) {
        task = 2;
        iter = iter + 1;
        summary.info = -2;
        break;
      } else {
        internalinfo = 0;
        col = 0;
        head = 0;
        theta = 1;
        iupdat = 0;
        updatd = false;
        break;
      }
    }
  }

  summary.num_iteration = iter;
  //summary.info = 0;

#ifdef USE_STREAM
  for (int i = 0; i < MAX_STREAM; i++) cudaStreamDestroy(streamPool[i]);
#endif

  memFree(workvec);
  memFree(workvec2);
  memFree(g);
  memFree(xold);
  memFree(xdiff);
  memFree(z);
  memFree(xp);
  memFree(zb);
  memFree(r);
  memFree(d);
  memFree(t);
  memFree(wa);
  memFree(buf_n_r);
  memFree(index);
  memFree(iorder);
  memFree(iwhere);
  memFree(temp_ind1);
  memFree(temp_ind2);
  memFree(temp_ind3);
  memFree(temp_ind4);

  // pitch = m
  memFree(ws);
  memFree(wy);
  memFree(sy);
  memFree(ss);
  memFree(yy);
  memFree(wt);
  memFree(workmat);

  // pitch = 2 * m
  memFree(wn);
  memFree(snd);
  memFree(buf_array_p);
#ifdef USE_STREAM
  memFree(buf_array_p1);
#endif
  memFree(buf_array_super);

  memFree(bufi_n_r);
  memFreeHost(sbgnrm_h);
  memFreeHost(dsave13);
}

template <typename real>
void lbfgsbactive(const int& n, const real* l, const real* u, const int* nbd,
                  real* x, int* iwhere) {
  active::prog0<real>(n, l, u, nbd, x, iwhere);
}

template <typename real>
void lbfgsbbmv(const int& m, const real* sy, real* wt, const int& col,
               const int& iPitch, const real* v, real* p,
               cublasHandle_t cublas_handle, const cudaStream_t& stream,
               int& info) {
  if (col == 0) {
    return;
  }

  bmv::prog0<real>(sy, col, iPitch, v, p, stream);

  bmv::prog1<real>(wt, col, iPitch, v, p, cublas_handle, stream);

  bmv::prog2<real>(sy, wt, col, iPitch, v, p, stream);
  info = 0;
}

template <typename real>
void lbfgsbcauchy(const int& n, const real* x, const real* l, const real* u,
                  const int* nbd, const real* g, real* t, real* xcp, real* xcpb,
                  const int& m, const real* wy, const real* ws, const real* sy,
                  const int iPitch, real* wt, const real& theta, const int& col,
                  const int& head, real* p, real* c, real* v, int& nint,
                  const real& sbgnrm, real* buf_s_r, real* buf_array_p,
                  int* iwhere, const int& iPitch_normal,
                  const real& machinemaximum, cublasHandle_t cublas_handle,
                  const cudaStream_t* streamPool, int& info) {
  info = 0;
  debugSync();
  cauchy::prog0<real>(n, x, l, u, nbd, g, t, xcp, xcpb, m, wy, ws, sy, iPitch,
                      wt, theta, col, head, p, c, v, nint, sbgnrm, buf_s_r,
                      buf_array_p, iwhere, iPitch_normal, machinemaximum,
                      cublas_handle, streamPool);
  debugSync();
}

template <typename real>
void lbfgsbcmprlb(const int& n, const int& m, const real* x, const real* g,
                  const real* ws, const real* wy, const real* sy, real* wt,
                  const real* z, real* r, real* wa, const int* index,
                  const real& theta, const int& col, const int& head,
                  const int& nfree, const bool& cnstnd, int& info,
                  real* workvec, real* workvec2, const int& iPitch,
                  cublasHandle_t cublas_handle, const cudaStream_t& stream) {
  debugSync();

  lbfgsbbmv<real>(m, sy, wt, col, iPitch, wa + 2 * m, wa, cublas_handle, stream,
                  info);
  cmprlb::prog1<real>(nfree, index, col, head, m, iPitch, wa, wy, ws, theta, z,
                      x, g, r, stream);
  debugSync();
}

template <typename real>
void lbfgsbformk(const int& n, const int& nsub, const int* ind,
                 const int& nenter, const int& ileave, const int* indx2,
                 const int& iupdat, const bool& updatd, real* wn, real* wn1,
                 const int& m, const real* ws, const real* wy, const real* sy,
                 const real& theta, const int& col, const int& head, int& info,
                 real* workvec, real* workmat, real* buf_array_p,
                 real* buf_array_super, const int& iPitch_wn,
                 const int& iPitch_ws, const int& iPitch_super,
                 const int& iPitch_normal, const real& machineepsilon,
                 cublasHandle_t cublas_handle, const cudaStream_t* streamPool) {
  int upcl = col;
  if (updatd) {
    if (iupdat > m) {
      debugSync();
      formk::prog0<real>(wn1, m, iPitch_wn, streamPool);
      debugSync();
    }

    int ipntr = head + col - 1;
    if (ipntr >= m) {
      ipntr = ipntr - m;
    }

    formk::prog1<real>(n, nsub, ipntr, ind, wn1, buf_array_p, ws, wy, head, m,
                       col, iPitch_ws, iPitch_wn, iPitch_normal, streamPool);

    if (n == nsub) formk::prog2<real>(wn1, col, m, iPitch_wn, streamPool);

    debugSync();

    int jy = col - 1;
    int jpntr = head + col - 1;
    if (jpntr >= m) {
      jpntr = jpntr - m;
    }

    formk::prog3<real>(ind, jpntr, head, m, col, n, nsub, iPitch_ws, iPitch_wn,
                       jy, ws, wy, buf_array_p, wn1, iPitch_normal, streamPool);
    debugSync();
    upcl = col - 1;
  } else {
    upcl = col;
  }

  if (upcl > 0) {
    formk::prog31<real>(indx2, head, m, upcl, col, nenter, ileave, n, iPitch_ws,
                        iPitch_wn, wy, buf_array_super, wn1, 1.0, iPitch_super,
                        streamPool);
    debugSync();

    formk::prog31<real>(indx2, head, m, upcl, col, nenter, ileave, n, iPitch_ws,
                        iPitch_wn, ws, buf_array_super, wn1 + m * iPitch_wn + m,
                        -1.0, iPitch_super, streamPool);
    debugSync();

    formk::prog32<real>(indx2, head, m, upcl, nenter, ileave, n, iPitch_ws,
                        iPitch_wn, wy, ws, buf_array_super, wn1, iPitch_super,
                        streamPool);
    debugSync();
  }
  formk::prog4<real>(col, iPitch_wn, iPitch_ws, m, wn1, theta, sy, wn,
                     streamPool);
  debugSync();

  dpofa::prog0<real>(wn, col, iPitch_wn, 0, machineepsilon, streamPool[2]);
  debugSync();

  formk::prog5<real>(col, iPitch_wn, wn, cublas_handle, streamPool);
  debugSync();

  dpofa::prog0<real>(wn, col, iPitch_wn, col, machineepsilon, streamPool[2]);
  debugSync();
}

template <typename real>
void lbfgsbformt(const int& m, real* wt, const real* sy, const real* ss,
                 const int& col, const real& theta, int& info,
                 const int& iPitch, const real& machineepsilon,
                 const cudaStream_t* streamPool) {
  debugSync();
  formt::prog01<real>(col, sy, ss, wt, iPitch, theta, streamPool[0]);
  debugSync();
  dpofa::prog0<real>(wt, col, iPitch, 0, machineepsilon, streamPool[0]);
  debugSync();

  info = 0;
}

template <typename real>
void lbfgsblnsrlb(const int& n, const real* l, const real* u, const int* nbd,
                  real* x, const real& f, real& fold, real& gd, real& gdold,
                  const real* g, real* d, real* r, real* t, const real* z,
                  real& stp, real& dnrm, real& dtd, real& xstep, real& stpmx,
                  const real& stpscaling, const int& iter, int& ifun,
                  int& iback, int& nfgv, int& info, int& task, int& csave,
                  int* isave, real* dsave, real* buf_s_r,
                  cublasHandle_t cublas_handle,
                  const cudaStream_t* streamPool) {
  int addinfo;

  addinfo = 0;
  const static real big = 1.0E10;
  const static real ftol = 1.0E-3;
  const static real gtol = 0.9E0;
  const static real xtol = 0.1E0;
  real* stpmx_host = NULL;
  real* stpmx_dev;

  if (task != 1) {
    cudaHostAlloc(&stpmx_host, sizeof(real), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&stpmx_dev, stpmx_host, 0);

    minimize::vdot_vv<real>(n, d, d, dtd, cublas_handle, streamPool[0]);

    *stpmx_host = stpmx = big;

    if (iter == 0) {
      stpmx = 1;
    } else {
      lnsrlb::prog0<real>(n, d, nbd, u, x, l, buf_s_r, stpmx_host, stpmx_dev,
                          streamPool[1]);
    }

    memCopyAsync(t, x, n * sizeof(real), cudaMemcpyDeviceToDevice,
                 streamPool[2]);
    memCopyAsync(r, g, n * sizeof(real), cudaMemcpyDeviceToDevice,
                 streamPool[3]);
    fold = f;
    ifun = 0;
    iback = 0;
    csave = 0;
  }
  minimize::vdot_vv<real>(n, g, d, gd, cublas_handle, streamPool[4]);

  cudaDeviceSynchronize();

  if (task != 1) {
    if (iter != 0) {
      stpmx = *stpmx_host;
      if (stpmx == big) stpmx = 1;
    }
    dnrm = sqrt(dtd);
    if (iter == 0) {
      stp = fmaxf(1.0, fminf(1.0 / dnrm, stpmx));
    } else {
      stp = 1;
    }
    cudaFreeHost(stpmx_host);
  }
  if (ifun == 0) {
    gdold = gd;
    if (gd >= 0) {
      info = -4;
      return;
    }
  }
  lbfgsbdcsrch<real>(f, gd, stp, ftol, gtol, xtol, real(0), stpmx, stpscaling,
                     csave, isave, dsave);
  if (csave != 3) {
    task = 1;
    ifun = ifun + 1;
    nfgv = nfgv + 1;
    iback = ifun - 1;
    if (stp == 1) {
      memCopyAsync(x, z, n * sizeof(real), cudaMemcpyDeviceToDevice,
                   streamPool[1]);
    } else {
      lnsrlb::prog2<real>(n, x, d, t, stp, streamPool[1]);
    }
  } else {
    task = 5;
  }

  xstep = stp * dnrm;
}

template <typename real>
void lbfgsbmatupdsub(const int& n, const int& m, real* wy, real* sy,
                     const real* r, const real* d, int& itail,
                     const int& iupdat, int& col, int& head, const real& dr,
                     const int& iPitch0, const int& iPitch_i,
                     const int& iPitch_j) {
  vmove_mv<real>(wy, r, itail, 0, n - 1, 0, iPitch0);
  for (int j = 0; j < col - 1; j++) {
    int pointr = Modular((head + j), m);
    sy[(col - 1) * iPitch_i + j * iPitch_j] =
        vdot_vm(d, wy, 0, n - 1, pointr, 0, iPitch0);
  }
  sy[(col - 1) * iPitch0 + col - 1] = dr;
}

template <typename real>
void lbfgsbmatupd(const int& n, const int& m, real* ws, real* wy, real* sy,
                  real* ss, const real* d, const real* r, int& itail,
                  const int& iupdat, int& col, int& head, real& theta,
                  const real& rr, const real& dr, const real& stp,
                  const real& dtd, const int& iPitch, real* buf_array_p,
                  const int& iPitch_normal, const cudaStream_t* streamPool) {
  if (iupdat <= m) {
    col = iupdat;
    itail = Modular((head + iupdat - 1), m);
  } else {
    itail = Modular(itail + 1, m);
    head = Modular(head + 1, m);
  }
  theta = rr / dr;

  matupd::prog0<real>(n, m, wy, sy, r, d, itail, iupdat, col, head, dr, iPitch,
                      iPitch, 1, buf_array_p, iPitch_normal, streamPool[1]);

  matupd::prog0<real>(n, m, ws, ss, d, d, itail, iupdat, col, head,
                      stp * stp * dtd, iPitch, 1, iPitch,
                      buf_array_p + m * iPitch_normal, iPitch_normal,
                      streamPool[2]);
}

template <typename real>
void lbfgsbprojgr(const int& n, const real* l, const real* u, const int* nbd,
                  const real* x, const real* g, real* buf_n, real* sbgnrm_h,
                  real* sbgnrm_d, const real& machinemaximum,
                  const cudaStream_t& stream) {
  projgr::prog0<real>(n, l, u, nbd, x, g, buf_n, sbgnrm_h, sbgnrm_d,
                      machinemaximum, stream);
}

template <typename real>
void lbfgsbsubsm(const int& n, const int& m, const int& nsub, const int* ind,
                 const real* l, const real* u, const int* nbd, real* x, real* d,
                 real* xp, const real* ws, const real* wy, const real& theta,
                 const real* xx, const real* gg, const int& col,
                 const int& head, real* wv, real* wn, int& info,
                 const int& iPitch_wn, const int& iPitch_ws, real* buf_array_p,
                 real* buf_s_r, int* bufi_s_r, const int& iPitch_normal,
                 cublasHandle_t cublas_handle, const cudaStream_t& stream) {
  subsm::prog0<real>(nsub, ind, head, m, col, iPitch_ws, buf_array_p, wy, ws, d,
                     wv, theta, iPitch_normal, stream);

  debugSync();
  subsm::prog1<real>(wn, col, iPitch_wn, wv, cublas_handle, stream);
  debugSync();

  debugSync();
  subsm::prog2<real>(nsub, ind, col, head, m, iPitch_ws, wv, wy, ws, theta, d,
                     stream);
  debugSync();

  cutilSafeCall(cudaMemcpyAsync(xp, x, n * sizeof(real),
                                cudaMemcpyDeviceToDevice, stream));

  real* pddp = NULL;
  real* pddp_dev = NULL;
  cudaMallocHost(&pddp, sizeof(real), cudaHostAllocMapped);
  cudaHostGetDevicePointer(&pddp_dev, pddp, 0);

  subsm::prog21<real>(n, nsub, ind, d, x, l, u, nbd, xx, gg, buf_s_r, pddp_dev,
                      stream);

  cutilSafeCall(cudaStreamSynchronize(stream));

  if (*pddp > 0) {
    cutilSafeCall(cudaMemcpyAsync(x, xp, n * sizeof(real),
                                  cudaMemcpyDeviceToDevice, stream));

    subsm::prog3<real>(nsub, ind, d, nbd, buf_s_r, bufi_s_r, x, u, l, stream);
    debugSync();
  }

  cudaFreeHost(pddp);
}

template <typename real>
void lbfgsbdcsrch(const real& f, const real& g, real& stp, const real& ftol,
                  const real& gtol, const real& xtol, const real& stpmin,
                  const real& stpmax, const real& stpscaling, int& task,
                  int* isave, real* dsave) {
  bool brackt;
  int stage;
  real finit;
  real ftest;
  real fm;
  real fx;
  real fxm;
  real fy;
  real fym;
  real ginit;
  real gtest;
  real gm;
  real gx;
  real gxm;
  real gy;
  real gym;
  real stx;
  real sty;
  real stmin;
  real stmax;
  real width;
  real width1;

  const static real xtrapl = 1.1E0;
  const static real xtrapu = 4.0E0;
  int counter = 0;
  while (true) {
    counter++;
    if (task == 0) {
      if (stp < stpmin || stp > stpmax || g >= 0 || ftol < 0 || gtol < 0 ||
          xtol < 0 || stpmin < 0 || stpmax < stpmin) {
        task = 2;
        return;
      }
      brackt = false;
      stage = 1;
      finit = f;
      ginit = g;
      gtest = ftol * ginit;
      width = stpmax - stpmin;
      width1 = width * 2.0;
      stx = 0;
      fx = finit;
      gx = ginit;
      sty = 0;
      fy = finit;
      gy = ginit;
      stmin = 0;
      stmax = stp + xtrapu * stp;
      task = 1;
      break;
    } else {
      brackt = (isave[0] == 1);
      stage = isave[1];
      ginit = dsave[0];
      gtest = dsave[1];
      gx = dsave[2];
      gy = dsave[3];
      finit = dsave[4];
      fx = dsave[5];
      fy = dsave[6];
      stx = dsave[7];
      sty = dsave[8];
      stmin = dsave[9];
      stmax = dsave[10];
      width = dsave[11];
      width1 = dsave[12];
    }
    ftest = finit + stp * gtest;
    if (stage == 1 && f <= ftest && g >= 0) {
      stage = 2;
    }
    if ((brackt &&
         (stp <= stmin || stp >= stmax || stmax - stmin <= xtol * stmax)) ||
        (stp == stpmax && f <= ftest && g <= gtest) ||
        (stp == stpmin && (f > ftest || g >= gtest)) ||
        (f <= ftest && fabs(g) <= gtol * (-ginit))) {
      task = 3;
      break;
    }
    if (stage == 1 && f <= fx && f > ftest) {
      fm = f - stp * gtest;
      fxm = fx - stx * gtest;
      fym = fy - sty * gtest;
      gm = g - gtest;
      gxm = gx - gtest;
      gym = gy - gtest;
      lbfgsbdcstep<real>(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt,
                         stmin, stmax);
      fx = fxm + stx * gtest;
      fy = fym + sty * gtest;
      gx = gxm + gtest;
      gy = gym + gtest;
    } else {
      lbfgsbdcstep<real>(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin,
                         stmax);
    }
    if (brackt) {
      if (fabs(sty - stx) >= 0.666666666666666667 * width1) {
        stp = stx + 0.5 * (sty - stx);
      }
      width1 = width;
      width = fabs(sty - stx);
      stmin = fminf(stx, sty);
      stmax = fmaxf(stx, sty);
    } else {
      stmin = stp + xtrapl * (stp - stx);
      stmax = stp + xtrapu * (stp - stx);
    }
    stp *= stpscaling;
    stp = fmaxf(stp, stpmin);
    stp = fminf(stp, stpmax);
    if (brackt && (stp <= stmin || stp >= stmax) ||
        brackt && stmax - stmin <= xtol * stmax) {
      stp = stx;
    }
    task = 1;
    break;
  }
  if (brackt) {
    isave[0] = 1;
  } else {
    isave[0] = 0;
  }
  isave[1] = stage;
  dsave[0] = ginit;
  dsave[1] = gtest;
  dsave[2] = gx;
  dsave[3] = gy;
  dsave[4] = finit;
  dsave[5] = fx;
  dsave[6] = fy;
  dsave[7] = stx;
  dsave[8] = sty;
  dsave[9] = stmin;
  dsave[10] = stmax;
  dsave[11] = width;
  dsave[12] = width1;
}

template <typename real>
void lbfgsbdcstep(real& stx, real& fx, real& dx, real& sty, real& fy, real& dy,
                  real& stp, const real& fp, const real& dp, bool& brackt,
                  const real& stpmin, const real& stpmax) {
  real gamma;
  real p;
  real q;
  real r;
  real s;
  real sgnd;
  real stpc;
  real stpf;
  real stpq;
  real theta;
  real stpstx;

  sgnd = dp * dx / fabs(dx);
  if (fp > fx) {
    theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
    s = fmaxf(fabs(theta), fmaxf(fabs(dx), fabs(dp)));
    gamma = s * sqrt((theta * theta) / (s * s) - dx / s * (dp / s));
    if (stp < stx) {
      gamma = -gamma;
    }
    p = gamma - dx + theta;
    q = gamma - dx + gamma + dp;
    r = p / q;
    stpstx = stp - stx;
    stpc = stx + r * stpstx;
    stpq = stx + dx / ((fx - fp) / stpstx + dx) * 0.5 * stpstx;
    if (fabs(stpc - stx) < fabs(stpq - stx)) {
      stpf = stpc;
    } else {
      stpf = stpc + (stpq - stpc) * 0.5;
    }
    brackt = true;
  } else if (sgnd < 0) {
    theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
    s = fmaxf(fabs(theta), fmaxf(fabs(dx), fabs(dp)));
    gamma = s * sqrt((theta * theta) / (s * s) - dx * dp / (s * s));
    if (stp > stx) {
      gamma = -gamma;
    }
    p = gamma - dp + theta;
    q = gamma * 2.0 - dp + dx;
    r = p / q;
    stpstx = stx - stp;
    stpc = stp + r * stpstx;
    stpq = stp + dp / (dp - dx) * stpstx;
    if (fabs(stpc - stp) > fabs(stpq - stp)) {
      stpf = stpc;
    } else {
      stpf = stpq;
    }
    brackt = true;
  } else if (fabs(dp) < fabs(dx)) {
    theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
    s = fmaxf(fabs(theta), fmaxf(fabs(dx), fabs(dp)));
    gamma = s * sqrt(fmaxf(0.0, (theta * theta) / (s * s) - dx * dp / (s * s)));
    if (stp > stx) {
      gamma = -gamma;
    }
    p = gamma - dp + theta;
    q = gamma + (dx - dp) + gamma;
    r = p / q;
    if (r < 0.0 && gamma != 0.0) {
      stpc = stp + r * (stx - stp);
    } else if (stp > stx) {
      stpc = stpmax;
    } else {
      stpc = stpmin;
    }
    stpq = stp + dp / (dp - dx) * (stx - stp);
    if (fabs(stpc - stp) < fabs(stpq - stp)) {
      stpf = stpc;
    } else {
      stpf = stpq;
    }
    if (brackt) {
      stpf = fmaxf(stp + 0.666666666666667 * (sty - stp), stpf);
    } else {
      stpf = fmaxf(stpmax, stpf);
    }
  } else if (brackt) {
    theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
    s = fmaxf(fabs(theta), fmaxf(fabs(dy), fabs(dp)));
    gamma = s * sqrt((theta * theta) / (s * s) - dy / s * (dp / s));
    if (stp > sty) {
      gamma = -gamma;
    }
    p = gamma - dp + theta;
    q = gamma - dp + gamma + dy;
    r = p / q;
    stpc = stp + r * (sty - stp);
    stpf = stpc;
  } else if (stp > stx) {
    stpf = stpmax;
  } else {
    stpf = stpmin;
  }

  if (fp > fx) {
    sty = stp;
    fy = fp;
    dy = dp;
  } else {
    if (sgnd < 0) {
      sty = stx;
      fy = fx;
      dy = dx;
    }
    stx = stp;
    fx = fp;
    dx = dp;
  }
  stp = stpf;
}

template <typename real>
bool lbfgsbdpofa(real* a, const int& n, const int& iPitch) {
  bool result;
  real s;
  real v;
  int j;
  int k;

  for (j = 0; j < n; j++) {
    s = 0.0;
    if (j >= 1) {
      for (k = 0; k <= j - 1; k++) {
        v = vdot_mm(a, a, k, 0, k - 1, iPitch, j, 0, iPitch);
        a[k * iPitch + j] = (a[k * iPitch + j] - v) / a[k * iPitch + k];
        s = s + a[k * iPitch + j] * a[k * iPitch + j];
      }
    }
    s = a[j * iPitch + j] - s;
    if (s <= 0.0) {
      result = false;
      return result;
    }
    a[j * iPitch + j] = sqrt(s);
  }
  result = true;
  return result;
}

template <typename real>
void lbfgsbdtrsl(real* t, const int& n, const int& iPitch, real* b,
                 const int& job, int& info) {
  real temp;
  real v;
  int cse;
  int j;
  int jj;

  for (j = 0; j < n; j++) {
    if (t[j * iPitch + j] == 0.0) {
      info = j;
      return;
    }
  }
  info = 0;
  cse = 1;
  if (job % 10 != 0) {
    cse = 2;
  }
  if (job % 100 / 10 != 0) {
    cse = cse + 2;
  }
  if (cse == 1) {
    b[0] = b[0] / t[0];
    if (n < 2) {
      return;
    }
    for (j = 1; j < n; j++) {
      temp = -b[j - 1];
      vadd_vm<real>(b, t, j, n - 1, j - 1, j, iPitch, temp);
      b[j] = b[j] / t[j * iPitch + j];
    }
    return;
  }
  if (cse == 2) {
    b[n - 1] = b[n - 1] / t[(n - 1) * iPitch + n - 1];
    if (n < 2) {
      return;
    }
    for (j = n - 2; j >= 0; j--) {
      temp = -b[j + 1];
      vadd_vm<real>(b, t, 0, j, j + 1, 0, iPitch, temp);
      b[j] = b[j] / t[j * iPitch + j];
    }
    return;
  }
  if (cse == 3) {
    b[n - 1] = b[n - 1] / t[(n - 1) * iPitch + n - 1];
    if (n < 2) {
      return;
    }
    for (jj = 2; jj <= n; jj++) {
      j = n - jj;
      v = vdot_vm<real>(b, t, j + 1, j + jj - 1, j, j + 1, iPitch);
      b[j] = b[j] - v;
      b[j] = b[j] / t[j * iPitch + j];
    }
    return;
  }
  if (cse == 4) {
    b[0] = b[0] / t[0];
    if (n < 2) {
      return;
    }
    for (j = 1; j < n; j++) {
      v = vdot_vm<real>(b, t, 0, j - 1, j, 0, iPitch);
      b[j] = b[j] - v;
      b[j] = b[j] / t[j * iPitch + j];
    }
    return;
  }
}

#define INST_HELPER(real)                                                      \
  template bool checkAvailabilty<real>(const LBFGSB_CUDA_STATE<real>&);        \
  template void lbfgsbminimize<real>(                                          \
      const int&, const LBFGSB_CUDA_STATE<real>&,                              \
      const LBFGSB_CUDA_OPTION<real>&, real*, const int*, const real*,         \
      const real*, LBFGSB_CUDA_SUMMARY<real>&);                                \
  template void lbfgsbactive<real>(const int&, const real*, const real*,       \
                                   const int*, real*, int*);                   \
  template void lbfgsbbmv<real>(const int&, const real*, real*, const int&,    \
                                const int&, const real*, real*,                \
                                cublasHandle_t, const cudaStream_t&, int&);    \
  template void lbfgsbcauchy<real>(                                            \
      const int&, const real*, const real*, const real*, const int*,           \
      const real*, real*, real*, real*, const int&, const real*, const real*,  \
      const real*, const int, real*, const real&, const int&, const int&,      \
      real*, real*, real*, int&, const real&, real*, real*, int*, const int&,  \
      const real&, cublasHandle_t, const cudaStream_t*, int&);                 \
  template void lbfgsbcmprlb<real>(                                            \
      const int&, const int&, const real*, const real*, const real*,           \
      const real*, const real*, real*, const real*, real*, real*, const int*,  \
      const real&, const int&, const int&, const int&, const bool&, int&,      \
      real*, real*, const int&, cublasHandle_t, const cudaStream_t&);          \
  template void lbfgsbformk<real>(                                             \
      const int&, const int&, const int*, const int&, const int&, const int*,  \
      const int&, const bool&, real*, real*, const int&, const real*,          \
      const real*, const real*, const real&, const int&, const int&, int&,     \
      real*, real*, real*, real*, const int&, const int&, const int&,          \
      const int&, const real&, cublasHandle_t, const cudaStream_t*);           \
  template void lbfgsbformt<real>(const int&, real*, const real*, const real*, \
                                  const int&, const real&, int&, const int&,   \
                                  const real&, const cudaStream_t*);           \
  template void lbfgsblnsrlb<real>(                                            \
      const int&, const real*, const real*, const int*, real*, const real&,    \
      real&, real&, real&, const real*, real*, real*, real*, const real*,      \
      real&, real&, real&, real&, real&, const real&, const int&, int&, int&,  \
      int&, int&, int&, int&, int*, real*, real*, cublasHandle_t,              \
      const cudaStream_t*);                                                    \
  template void lbfgsbmatupdsub<real>(const int&, const int&, real*, real*,    \
                                      const real*, const real*, int&,          \
                                      const int&, int&, int&, const real&,     \
                                      const int&, const int&, const int&);     \
  template void lbfgsbmatupd<real>(                                            \
      const int&, const int&, real*, real*, real*, real*, const real*,         \
      const real*, int&, const int&, int&, int&, real&, const real&,           \
      const real&, const real&, const real&, const int&, real*, const int&,    \
      const cudaStream_t*);                                                    \
  template void lbfgsbprojgr<real>(                                            \
      const int&, const real*, const real*, const int*, const real*,           \
      const real*, real*, real*, real*, const real&, const cudaStream_t&);     \
  template void lbfgsbsubsm<real>(                                             \
      const int&, const int&, const int&, const int*, const real*,             \
      const real*, const int*, real*, real*, real*, const real*, const real*,  \
      const real&, const real*, const real*, const int&, const int&, real*,    \
      real*, int&, const int&, const int&, real*, real*, int*, const int&,     \
      cublasHandle_t, const cudaStream_t&);                                    \
  template void lbfgsbdcsrch<real>(                                            \
      const real&, const real&, real&, const real&, const real&, const real&,  \
      const real&, const real&, const real&, int&, int*, real*);               \
  template void lbfgsbdcstep<real>(real&, real&, real&, real&, real&, real&,   \
                                   real&, const real&, const real&, bool&,     \
                                   const real&, const real&);                  \
  template bool lbfgsbdpofa<real>(real*, const int&, const int&);              \
  template void lbfgsbdtrsl<real>(real*, const int&, const int&, real*,        \
                                  const int&, int&);

INST_HELPER(double);
INST_HELPER(float);
};  // namespace cuda
}  // namespace lbfgsbcuda

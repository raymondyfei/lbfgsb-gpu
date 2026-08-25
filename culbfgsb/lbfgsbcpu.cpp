/*************************************************************************
NEOS, November 1994. (Latest revision June 1996.)
Optimization Technology Center.
Argonne National Laboratory and Northwestern University.

Written by Ciyou Zhu in collaboration with
R.H. Byrd, P. Lu-Chen and J. Nocedal.

Contributors:
    * Sergey Bochkanov (ALGLIB project). Translation from FORTRAN to
      pseudocode.

This software is freely available, but we  expect  that  all  publications
describing  work using this software, or all commercial products using it,
quote at least one of the references given below:
    * R. H. Byrd, P. Lu and J. Nocedal.  A Limited  Memory  Algorithm  for
      Bound Constrained Optimization, (1995), SIAM Journal  on  Scientific
      and Statistical Computing , 16, 5, pp. 1190-1208.
    * C. Zhu, R.H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for  large  scale  bound  constrained  optimization
      (1997), ACM Transactions on Mathematical Software,  Vol 23,  Num. 4,
      pp. 550 - 560.
*************************************************************************/

#include "culbfgsb/lbfgsbcpu.h"

/*************************************************************************
The  subroutine  minimizes  the  function  F(x) of N arguments with simple
constraints using a quasi-Newton method (LBFGS scheme) which is  optimized
to use a minimum amount of memory.

The subroutine generates the approximation of an inverse Hessian matrix by
using information about the last M steps of the algorithm (instead  of N).
It lessens a required amount of memory from a value  of  order  N^2  to  a
value of order 2*N*M.

This subroutine uses the FuncGrad subroutine which calculates the value of
the function F and gradient G in point X. The programmer should define the
FuncGrad subroutine by himself.  It should be noted  that  the  subroutine
doesn't need to waste  time for memory allocation of array G, because  the
memory is allocated in calling the  subroutine.  Setting  a  dimension  of
array G each time when calling a subroutine will excessively slow down  an
algorithm.

The programmer could also redefine the LBFGSNewIteration subroutine  which
is called on each new step. The current point X, the function value F  and
the gradient G are passed  into  this  subroutine.  It  is  reasonable  to
redefine the subroutine for better debugging, for  example,  to  visualize
the solution process.

Input parameters:
    N       -   problem dimension. N>0
    M       -   number of  corrections  in  the  BFGS  scheme  of  Hessian
                approximation  update.  Recommended value:  3<=M<=7.   The
                smaller value causes worse convergence,  the  bigger  will
                not  cause  a  considerably  better  convergence, but will
                cause a fall in the performance. M<=N.
    X       -   initial solution approximation.
                Array whose index ranges from 1 to N.
    EpsG    -   positive number which defines a precision of  search.  The
                subroutine finishes its work if the condition ||G|| < EpsG
                is satisfied, where ||.|| means Euclidian norm, G - gradient
                projection onto a feasible set, X - current approximation.
    EpsF    -   positive number which defines a precision of  search.  The
                subroutine  finishes  its  work if on iteration number k+1
                the condition |F(k+1)-F(k)| <= EpsF*max{|F(k)|, |F(k+1)|, 1}
                is satisfied.
    EpsX    -   positive number which defines a precision of  search.  The
                subroutine  finishes  its  work if on iteration number k+1
                the condition |X(k+1)-X(k)| <= EpsX is satisfied.
    MaxIts  -   maximum number of iterations.
                If MaxIts=0, the number of iterations is unlimited.
    NBD     -   constraint type. If NBD(i) is equal to:
                * 0, X(i) has no constraints,
                * 1, X(i) has only lower boundary,
                * 2, X(i) has both lower and upper boundaries,
                * 3, X(i) has only upper boundary,
                Array whose index ranges from 1 to N.
    L       -   lower boundaries of X(i) variables.
                Array whose index ranges from 1 to N.
    U       -   upper boundaries of X(i) variables.
                Array whose index ranges from 1 to N.

Output parameters:
    X       -   solution approximation.
Array whose index ranges from 1 to N.
    Info    -   a return code:
                    * -2 unknown internal error,
                    * -1 wrong parameters were specified,
                    * 0 interrupted by user,
                    * 1 relative function decreasing is less or equal to EpsF,
                    * 2 step is less or equal to EpsX,
                    * 4 gradient norm is less or equal to EpsG,
                    * 5 number of iterations exceeds MaxIts.

FuncGrad routine description. User-defined.
Input parameters:
    X   -   array whose index ranges from 1 to N.
Output parameters:
    F   -   function value at X.
    G   -   function gradient.
            Array whose index ranges from 1 to N.
The memory for array G has already been allocated in the calling subroutine,
and it isn't necessary to allocate it in the FuncGrad subroutine.

    NEOS, November 1994. (Latest revision June 1996.)
    Optimization Technology Center.
    Argonne National Laboratory and Northwestern University.

    Written by Ciyou Zhu in collaboration with
    R.H. Byrd, P. Lu-Chen and J. Nocedal.
*************************************************************************/
using namespace lbfgsbcuda::cpu;

namespace lbfgsbcuda {
namespace cpu {
template <typename real>
void lbfgsbminimize(const int& n, const LBFGSB_CUDA_STATE<real>& state,
                    const LBFGSB_CUDA_OPTION<real>& option, real* x_,
                    const int* nbd_, const real* l_, const real* u_,
                    LBFGSB_CUDA_SUMMARY<real>& summary) {
  ap::template_1d_array<real, true> x;
  ap::integer_1d_array nbd;
  ap::template_1d_array<real, true> l;
  ap::template_1d_array<real, true> u;

  x.setcontent(1, n, x_);
  nbd.setcontent(1, n, nbd_);
  l.setcontent(1, n, l_);
  u.setcontent(1, n, u_);

  real f;
  ap::template_1d_array<real, true> g;
  ap::template_1d_array<real, true> xold;
  ap::template_1d_array<real, true> xdiff;
  ap::template_2d_array<real, true> ws;
  ap::template_2d_array<real, true> wy;
  ap::template_2d_array<real, true> sy;
  ap::template_2d_array<real, true> ss;
  ap::template_2d_array<real, true> yy;
  ap::template_2d_array<real, true> wt;
  ap::template_2d_array<real, true> wn;
  ap::template_2d_array<real, true> snd;
  ap::template_1d_array<real, true> z;
  ap::template_1d_array<real, true> r;
  ap::template_1d_array<real, true> d;
  ap::template_1d_array<real, true> t;
  ap::template_1d_array<real, true> wa;
  ap::template_1d_array<real, true> sg;
  ap::template_1d_array<real, true> sgo;
  ap::template_1d_array<real, true> yg;
  ap::template_1d_array<real, true> ygo;
  ap::integer_1d_array index;
  ap::integer_1d_array iwhere;
  ap::integer_1d_array indx2;
  int csave;
  ap::boolean_1d_array lsave;
  ap::integer_1d_array isave;
  ap::template_1d_array<real, true> dsave;
  int task;
  bool prjctd;
  bool cnstnd;
  bool boxed;
  bool updatd;
  bool wrk;
  int i;
  int k;
  int nintol;
  int iback;
  int nskip;
  int head;
  int col;
  int iter;
  int itail;
  int iupdat;
  int nint;
  int nfgv;
  int internalinfo;
  int ifun;
  int iword;
  int nfree;
  int nact;
  int ileave;
  int nenter;
  real theta;
  real fold;
  real dr;
  real rr;
  real dnrm;
  real xstep;
  real sbgnrm;
  real ddum;
  real dtd;
  real gd;
  real gdold;
  real stp;
  real stpmx;
  real tf;
  ap::template_1d_array<real, true> workvec;
  ap::template_1d_array<real, true> workvec2;
  ap::template_1d_array<real, true> dsave13;
  ap::template_1d_array<real, true> wa0;
  ap::template_1d_array<real, true> wa1;
  ap::template_1d_array<real, true> wa2;
  ap::template_1d_array<real, true> wa3;
  ap::template_2d_array<real, true> workmat;
  ap::integer_1d_array isave2;

  int m = std::min(option.hessian_approximate_dimension, 8);

  workvec.setbounds(1, m);
  workvec2.setbounds(1, 2 * m);
  workmat.setbounds(1, m, 1, m);
  isave2.setbounds(1, 2);
  dsave13.setbounds(1, 13);
  wa0.setbounds(1, 2 * m);
  wa1.setbounds(1, 2 * m);
  wa2.setbounds(1, 2 * m);
  wa3.setbounds(1, 2 * m);
  g.setbounds(1, n);
  xold.setbounds(1, n);
  xdiff.setbounds(1, n);
  ws.setbounds(1, n, 1, m);
  wy.setbounds(1, n, 1, m);
  sy.setbounds(1, m, 1, m);
  ss.setbounds(1, m, 1, m);
  yy.setbounds(1, m, 1, m);
  wt.setbounds(1, m, 1, m);
  wn.setbounds(1, 2 * m, 1, 2 * m);
  snd.setbounds(1, 2 * m, 1, 2 * m);
  z.setbounds(1, n);
  r.setbounds(1, n);
  d.setbounds(1, n);
  t.setbounds(1, n);
  wa.setbounds(1, 8 * m);
  sg.setbounds(1, m);
  sgo.setbounds(1, m);
  yg.setbounds(1, m);
  ygo.setbounds(1, m);
  index.setbounds(1, n);
  iwhere.setbounds(1, n);
  indx2.setbounds(1, n);
  lsave.setbounds(1, 4);
  isave.setbounds(1, 23);
  dsave.setbounds(1, 29);
  col = 0;
  head = 1;
  theta = 1;
  iupdat = 0;
  updatd = false;
  iter = 0;
  nfgv = 0;
  nint = 0;
  nintol = 0;
  nskip = 0;
  nfree = n;
  internalinfo = 0;
  lbfgsberrclb<real>(n, m, option.eps_f, l, u, nbd, task, internalinfo, k);
  if (task == 2 || option.max_iteration < 0 || option.eps_g < 0 ||
      option.eps_x < 0) {
    summary.info = -1;
    memcpy(x_, x.getcontent(), n * sizeof(real));
    return;
  }
  lbfgsbactive<real>(n, l, u, nbd, x, iwhere, prjctd, cnstnd, boxed);
  ap::vmove(&xold(1), &x(1), ap::vlen(1, n));
  cudaStream_t cunullptr = NULL;
  if (state.m_funcgrad_callback)
    state.m_funcgrad_callback(x.getcontent(), f, g.getcontent(), cunullptr,
                              summary);
  nfgv = 1;
  lbfgsbprojgr<real>(n, l, u, nbd, x, g, sbgnrm);
  summary.residual_g = sbgnrm;
  if (sbgnrm <= option.eps_g) {
    summary.info = 4;
    memcpy(x_, x.getcontent(), n * sizeof(real));
    return;
  }
  while (true) {
    iword = -1;
    if (!cnstnd && col > 0) {
      ap::vmove(&z(1), &x(1), ap::vlen(1, n));
      wrk = updatd;
      nint = 0;
    } else {
      ap::vmove(&wa0(1), &wa(1), ap::vlen(1, 2 * m));
      ap::vmove(&wa1(1), &wa(2 * m + 1), ap::vlen(1, 2 * m));
      ap::vmove(&wa2(1), &wa(4 * m + 1), ap::vlen(1, 2 * m));
      ap::vmove(&wa3(1), &wa(6 * m + 1), ap::vlen(1, 2 * m));
      lbfgsbcauchy<real>(n, x, l, u, nbd, g, indx2, iwhere, t, d, z, m, wy, ws,
                         sy, wt, theta, col, head, wa0, wa1, wa2, wa3, nint, sg,
                         yg, sbgnrm, internalinfo, workvec);
      ap::vmove(&wa(1), &wa0(1), ap::vlen(1, 2 * m));
      ap::vmove(&wa(2 * m + 1), &wa1(1), ap::vlen(2 * m + 1, 4 * m));
      ap::vmove(&wa(4 * m + 1), &wa2(1), ap::vlen(4 * m + 1, 6 * m));
      ap::vmove(&wa(6 * m + 1), &wa3(1), ap::vlen(6 * m + 1, 8 * m));
      if (internalinfo != 0) {
        internalinfo = 0;
        col = 0;
        head = 1;
        theta = 1;
        iupdat = 0;
        updatd = false;
        continue;
      }
      nintol = nintol + nint;
      lbfgsbfreev(n, nfree, index, nenter, ileave, indx2, iwhere, wrk, updatd,
                  cnstnd, iter);
      nact = n - nfree;
    }
    if (nfree != 0 && col != 0) {
      if (wrk) {
        lbfgsbformk<real>(n, nfree, index, nenter, ileave, indx2, iupdat,
                          updatd, wn, snd, m, ws, wy, sy, theta, col, head,
                          internalinfo, workvec, workmat);
      }
      if (internalinfo != 0) {
        internalinfo = 0;
        col = 0;
        head = 1;
        theta = 1;
        iupdat = 0;
        updatd = false;
        continue;
      }
      lbfgsbcmprlb<real>(n, m, x, g, ws, wy, sy, wt, z, r, wa, index, theta,
                         col, head, nfree, cnstnd, internalinfo, workvec,
                         workvec2);
      if (internalinfo == 0) {
        lbfgsbsubsm<real>(n, m, nfree, index, l, u, nbd, z, r, ws, wy, theta,
                          col, head, iword, wa, wn, internalinfo);
      }
      if (internalinfo != 0) {
        internalinfo = 0;
        col = 0;
        head = 1;
        theta = 1;
        iupdat = 0;
        updatd = false;
        continue;
      }
    }
    for (i = 1; i <= n; i++) {
      d(i) = z(i) - x(i);
    }
    task = 0;
    while (true) {
      lbfgsblnsrlb<real>(n, l, u, nbd, x, f, fold, gd, gdold, g, d, r, t, z,
                         stp, dnrm, dtd, xstep, stpmx, iter, ifun, iback, nfgv,
                         internalinfo, task, boxed, cnstnd, csave, isave2,
                         dsave13);
      if (internalinfo != 0 || iback >= 20 || task != 1) {
        break;
      }
      if (state.m_funcgrad_callback)
        state.m_funcgrad_callback(x.getcontent(), f, g.getcontent(), cunullptr,
                                  summary);
    }
    if (internalinfo != 0) {
      ap::vmove(&x(1), &t(1), ap::vlen(1, n));
      ap::vmove(&g(1), &r(1), ap::vlen(1, n));
      f = fold;
      if (col == 0) {
        if (internalinfo == 0) {
          internalinfo = -9;
          nfgv = nfgv - 1;
          ifun = ifun - 1;
          iback = iback - 1;
        }
        task = 2;
        iter = iter + 1;
        summary.info = -2;
        memcpy(x_, x.getcontent(), n * sizeof(real));
        return;
      } else {
        if (internalinfo == 0) {
          nfgv = nfgv - 1;
        }
        internalinfo = 0;
        col = 0;
        head = 1;
        theta = 1;
        iupdat = 0;
        updatd = false;
        continue;
      }
    }
    iter = iter + 1;
    summary.num_iteration = iter;
    cudaStream_t cudanullptr = NULL;
    if (state.m_after_iteration_callback)
      state.m_after_iteration_callback(x.getcontent(), f, g.getcontent(),
                                       cudanullptr, summary);
    lbfgsbprojgr<real>(n, l, u, nbd, x, g, sbgnrm);
    summary.residual_g = sbgnrm;
    if (sbgnrm <= option.eps_g) {
      summary.info = 4;
      memcpy(x_, x.getcontent(), n * sizeof(real));
      return;
    }
    ap::vmove(&xdiff(1), &xold(1), ap::vlen(1, n));
    ap::vsub(&xdiff(1), &x(1), ap::vlen(1, n));
    tf = ap::vdotproduct(&xdiff(1), &xdiff(1), ap::vlen(1, n));
    tf = sqrt(tf);
    summary.residual_x = tf;
    if (tf <= option.eps_x) {
      summary.info = 2;
      memcpy(x_, x.getcontent(), n * sizeof(real));
      return;
    }
    ddum = ap::maxreal<real>(fabs(fold), ap::maxreal<real>(fabs(f), real(1)));
    summary.residual_f = (fold - f) / ddum;
    if (fold - f <= option.eps_f * ddum) {
      summary.info = 1;
      memcpy(x_, x.getcontent(), n * sizeof(real));
      return;
    }
    if (iter > option.max_iteration && option.max_iteration > 0) {
      summary.info = 5;
      memcpy(x_, x.getcontent(), n * sizeof(real));
      return;
    }
    if (state.m_customized_stopping_callback &&
        state.m_customized_stopping_callback(x.getcontent(), f, g.getcontent(),
                                             cudanullptr, summary)) {
      summary.info = 0;
      memcpy(x_, x.getcontent(), n * sizeof(real));
      return;
    }
    ap::vmove(&xold(1), &x(1), ap::vlen(1, n));
    for (i = 1; i <= n; i++) {
      r(i) = g(i) - r(i);
    }
    rr = ap::vdotproduct(&r(1), &r(1), ap::vlen(1, n));
    if (stp == 1) {
      dr = gd - gdold;
      ddum = -gdold;
    } else {
      dr = (gd - gdold) * stp;
      ap::vmul(&d(1), ap::vlen(1, n), stp);
      ddum = -gdold * stp;
    }
    if (dr <= ap::machineepsilon * ddum) {
      nskip = nskip + 1;
      updatd = false;
    } else {
      updatd = true;
      iupdat = iupdat + 1;
      lbfgsbmatupd<real>(n, m, ws, wy, sy, ss, d, r, itail, iupdat, col, head,
                         theta, rr, dr, stp, dtd);
      lbfgsbformt<real>(m, wt, sy, ss, col, theta, internalinfo);
      if (internalinfo != 0) {
        internalinfo = 0;
        col = 0;
        head = 1;
        theta = 1;
        iupdat = 0;
        updatd = false;
        continue;
      }
    }
  }

  memcpy(x_, x.getcontent(), n * sizeof(real));
}

template <typename real>
void lbfgsbactive(const int& n, const ap::template_1d_array<real, true>& l,
                  const ap::template_1d_array<real, true>& u,
                  const ap::integer_1d_array& nbd,
                  ap::template_1d_array<real, true>& x,
                  ap::integer_1d_array& iwhere, bool& prjctd, bool& cnstnd,
                  bool& boxed) {
  int nbdd;
  int i;

  nbdd = 0;
  prjctd = false;
  cnstnd = false;
  boxed = true;
  for (i = 1; i <= n; i++) {
    if (nbd(i) > 0) {
      if (nbd(i) <= 2 && x(i) <= l(i)) {
        if (x(i) < l(i)) {
          prjctd = true;
          x(i) = l(i);
        }
        nbdd = nbdd + 1;
      } else {
        if (nbd(i) >= 2 && x(i) >= u(i)) {
          if (x(i) > u(i)) {
            prjctd = true;
            x(i) = u(i);
          }
          nbdd = nbdd + 1;
        }
      }
    }
  }
  for (i = 1; i <= n; i++) {
    if (nbd(i) != 2) {
      boxed = false;
    }
    if (nbd(i) == 0) {
      iwhere(i) = -1;
    } else {
      cnstnd = true;
      if (nbd(i) == 2 && u(i) - l(i) <= 0) {
        iwhere(i) = 3;
      } else {
        iwhere(i) = 0;
      }
    }
  }
}

template <typename real>
void lbfgsbbmv(const int& m, const ap::template_2d_array<real, true>& sy,
               ap::template_2d_array<real, true>& wt, const int& col,
               const ap::template_1d_array<real, true>& v,
               ap::template_1d_array<real, true>& p, int& info,
               ap::template_1d_array<real, true>& workvec) {
  int i;
  int k;
  int i2;
  real s;

  if (col == 0) {
    return;
  }
  p(col + 1) = v(col + 1);
  for (i = 2; i <= col; i++) {
    i2 = col + i;
    s = 0.0;
    for (k = 1; k <= i - 1; k++) {
      s = s + sy(i, k) * v(k) / sy(k, k);
    }
    p(i2) = v(i2) + s;
  }
  ap::vmove(&workvec(1), &p(col + 1), ap::vlen(1, col));
  lbfgsbdtrsl(wt, col, workvec, 11, info);
  ap::vmove(&p(col + 1), &workvec(1), ap::vlen(col + 1, col + col));
  if (info != 0) {
    return;
  }
  for (i = 1; i <= col; i++) {
    p(i) = v(i) / sqrt(sy(i, i));
  }
  ap::vmove(&workvec(1), &p(col + 1), ap::vlen(1, col));
  lbfgsbdtrsl(wt, col, workvec, 1, info);
  ap::vmove(&p(col + 1), &workvec(1), ap::vlen(col + 1, col + col));
  if (info != 0) {
    return;
  }
  for (i = 1; i <= col; i++) {
    p(i) = -p(i) / sqrt(sy(i, i));
  }
  for (i = 1; i <= col; i++) {
    s = 0;
    for (k = i + 1; k <= col; k++) {
      s = s + sy(k, i) * p(col + k) / sy(i, i);
    }
    p(i) = p(i) + s;
  }
}

template <typename real>
void lbfgsbcauchy(
    const int& n, const ap::template_1d_array<real, true>& x,
    const ap::template_1d_array<real, true>& l,
    const ap::template_1d_array<real, true>& u, const ap::integer_1d_array& nbd,
    const ap::template_1d_array<real, true>& g, ap::integer_1d_array& iorder,
    ap::integer_1d_array& iwhere, ap::template_1d_array<real, true>& t,
    ap::template_1d_array<real, true>& d,
    ap::template_1d_array<real, true>& xcp, const int& m,
    const ap::template_2d_array<real, true>& wy,
    const ap::template_2d_array<real, true>& ws,
    const ap::template_2d_array<real, true>& sy,
    ap::template_2d_array<real, true>& wt, const real& theta, const int& col,
    const int& head, ap::template_1d_array<real, true>& p,
    ap::template_1d_array<real, true>& c,
    ap::template_1d_array<real, true>& wbp,
    ap::template_1d_array<real, true>& v, int& nint,
    const ap::template_1d_array<real, true>& sg,
    const ap::template_1d_array<real, true>& yg, const real& sbgnrm, int& info,
    ap::template_1d_array<real, true>& workvec) {
  bool xlower;
  bool xupper;
  bool bnded;
  int i;
  int j;
  int col2;
  int nfree;
  int nbreak;
  int pointr;
  int ibp;
  int nleft;
  int ibkmin;
  int iter;
  real f1;
  real f2;
  real dt;
  real dtm;
  real tsum;
  real dibp;
  real zibp;
  real dibp2;
  real bkmin;
  real tu;
  real tl;
  real wmc;
  real wmp;
  real wmw;
  real tj;
  real tj0;
  real neggi;
  real f2org;
  real tmpv;

  if (sbgnrm <= 0) {
    ap::vmove<real>(&xcp(1), &x(1), ap::vlen(1, n));
    return;
  }
  bnded = true;
  nfree = n + 1;
  nbreak = 0;
  ibkmin = 0;
  bkmin = 0;
  col2 = 2 * col;
  f1 = 0;
  for (i = 1; i <= col2; i++) {
    p(i) = 0;
  }
  for (i = 1; i <= n; i++) {
    neggi = -g(i);
    if (iwhere(i) != 3 && iwhere(i) != -1) {
      tl = 0;
      tu = 0;
      if (nbd(i) <= 2) {
        tl = x(i) - l(i);
      }
      if (nbd(i) >= 2) {
        tu = u(i) - x(i);
      }
      xlower = nbd(i) <= 2 && tl <= 0;
      xupper = nbd(i) >= 2 && tu <= 0;
      iwhere(i) = 0;
      if (xlower) {
        if (neggi <= 0) {
          iwhere(i) = 1;
        }
      } else {
        if (xupper) {
          if (neggi >= 0) {
            iwhere(i) = 2;
          }
        } else {
          if (fabs(neggi) <= 0) {
            iwhere(i) = -3;
          }
        }
      }
    }
    pointr = head;
    if (iwhere(i) != 0 && iwhere(i) != -1) {
      d(i) = 0;
    } else {
      d(i) = neggi;
      f1 = f1 - neggi * neggi;
      for (j = 1; j <= col; j++) {
        p(j) = p(j) + wy(i, pointr) * neggi;
        p(col + j) = p(col + j) + ws(i, pointr) * neggi;
        pointr = pointr % m + 1;
      }
      if (nbd(i) <= 2 && nbd(i) != 0 && neggi < 0) {
        nbreak = nbreak + 1;
        iorder(nbreak) = i;
        t(nbreak) = tl / (-neggi);
        if (nbreak == 1 || t(nbreak) < bkmin) {
          bkmin = t(nbreak);
          ibkmin = nbreak;
        }
      } else {
        if (nbd(i) >= 2 && neggi > 0) {
          nbreak = nbreak + 1;
          iorder(nbreak) = i;
          t(nbreak) = tu / neggi;
          if (nbreak == 1 || t(nbreak) < bkmin) {
            bkmin = t(nbreak);
            ibkmin = nbreak;
          }
        } else {
          nfree = nfree - 1;
          iorder(nfree) = i;
          if (fabs(neggi) > 0) {
            bnded = false;
          }
        }
      }
    }
  }
  if (theta != 1) {
    ap::vmul<real>(&p(col + 1), ap::vlen(col + 1, col + col), theta);
  }
  ap::vmove<real>(&xcp(1), &x(1), ap::vlen(1, n));
  if (nbreak == 0 && nfree == n + 1) {
    return;
  }
  for (j = 1; j <= col2; j++) {
    c(j) = 0;
  }
  f2 = -theta * f1;
  f2org = f2;
  if (col > 0) {
    lbfgsbbmv<real>(m, sy, wt, col, p, v, info, workvec);
    if (info != 0) {
      return;
    }
    tmpv = ap::vdotproduct<real>(&v(1), &p(1), ap::vlen(1, col2));
    f2 = f2 - tmpv;
  }
  dtm = -f1 / f2;
  tsum = 0;
  nint = 1;
  if (nbreak != 0) {
    nleft = nbreak;
    iter = 1;
    tj = 0;
    while (true) {
      tj0 = tj;
      if (iter == 1) {
        tj = bkmin;
        ibp = iorder(ibkmin);
      } else {
        if (iter == 2) {
          if (ibkmin != nbreak) {
            t(ibkmin) = t(nbreak);
            iorder(ibkmin) = iorder(nbreak);
          }
        }
        lbfgsbhpsolb<real>(nleft, t, iorder, iter - 2);
        tj = t(nleft);
        ibp = iorder(nleft);
      }
      dt = tj - tj0;
      if (dtm < dt) {
        break;
      }
      tsum = tsum + dt;
      nleft = nleft - 1;
      iter = iter + 1;
      dibp = d(ibp);
      d(ibp) = 0;
      if (dibp > 0) {
        zibp = u(ibp) - x(ibp);
        xcp(ibp) = u(ibp);
        iwhere(ibp) = 2;
      } else {
        zibp = l(ibp) - x(ibp);
        xcp(ibp) = l(ibp);
        iwhere(ibp) = 1;
      }
      if (nleft == 0 && nbreak == n) {
        dtm = dt;
        if (col > 0) {
          ap::vadd<real>(&c(1), &p(1), ap::vlen(1, col2), dtm);
        }
        return;
      }
      nint = nint + 1;
      dibp2 = ap::sqr(dibp);
      f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
      f2 = f2 - theta * dibp2;
      if (col > 0) {
        ap::vadd<real>(&c(1), &p(1), ap::vlen(1, col2), dt);
        pointr = head;
        for (j = 1; j <= col; j++) {
          wbp(j) = wy(ibp, pointr);
          wbp(col + j) = theta * ws(ibp, pointr);
          pointr = pointr % m + 1;
        }
        lbfgsbbmv<real>(m, sy, wt, col, wbp, v, info, workvec);
        if (info != 0) {
          return;
        }
        wmc = ap::vdotproduct<real>(&c(1), &v(1), ap::vlen(1, col2));
        wmp = ap::vdotproduct<real>(&p(1), &v(1), ap::vlen(1, col2));
        wmw = ap::vdotproduct<real>(&wbp(1), &v(1), ap::vlen(1, col2));
        ap::vsub<real>(&p(1), &wbp(1), ap::vlen(1, col2), dibp);
        f1 = f1 + dibp * wmc;
        f2 = f2 + static_cast<real>(2.0) * dibp * wmp - dibp2 * wmw;
      }
      f2 = ap::maxreal<real>(ap::machineepsilon * f2org, f2);
      if (nleft > 0) {
        dtm = -f1 / f2;
        continue;
      } else {
        if (bnded) {
          f1 = 0;
          f2 = 0;
          dtm = 0;
        } else {
          dtm = -f1 / f2;
        }
      }
      break;
    }
  }
  if (dtm <= 0) {
    dtm = 0;
  }
  tsum = tsum + dtm;
  ap::vadd<real>(&xcp(1), &d(1), ap::vlen(1, n), tsum);
  if (col > 0) {
    ap::vadd<real>(&c(1), &p(1), ap::vlen(1, col2), dtm);
  }
}

template <typename real>
void lbfgsbcmprlb(const int& n, const int& m,
                  const ap::template_1d_array<real, true>& x,
                  const ap::template_1d_array<real, true>& g,
                  const ap::template_2d_array<real, true>& ws,
                  const ap::template_2d_array<real, true>& wy,
                  const ap::template_2d_array<real, true>& sy,
                  ap::template_2d_array<real, true>& wt,
                  const ap::template_1d_array<real, true>& z,
                  ap::template_1d_array<real, true>& r,
                  ap::template_1d_array<real, true>& wa,
                  const ap::integer_1d_array& index, const real& theta,
                  const int& col, const int& head, const int& nfree,
                  const bool& cnstnd, int& info,
                  ap::template_1d_array<real, true>& workvec,
                  ap::template_1d_array<real, true>& workvec2) {
  int i;
  int j;
  int k;
  int pointr;
  real a1;
  real a2;

  if (!cnstnd && col > 0) {
    for (i = 1; i <= n; i++) {
      r(i) = -g(i);
    }
  } else {
    for (i = 1; i <= nfree; i++) {
      k = index(i);
      r(i) = -theta * (z(k) - x(k)) - g(k);
    }
    ap::vmove<real>(&workvec2(1), &wa(2 * m + 1), ap::vlen(1, 2 * m));
    lbfgsbbmv<real>(m, sy, wt, col, workvec2, wa, info, workvec);
    ap::vmove<real>(&wa(2 * m + 1), &workvec2(1), ap::vlen(2 * m + 1, 4 * m));
    if (info != 0) {
      info = -8;
      return;
    }
    pointr = head;
    for (j = 1; j <= col; j++) {
      a1 = wa(j);
      a2 = theta * wa(col + j);
      for (i = 1; i <= nfree; i++) {
        k = index(i);
        r(i) = r(i) + wy(k, pointr) * a1 + ws(k, pointr) * a2;
      }
      pointr = pointr % m + 1;
    }
  }
}

template <typename real>
void lbfgsberrclb(const int& n, const int& m, const real& factr,
                  const ap::template_1d_array<real, true>& l,
                  const ap::template_1d_array<real, true>& u,
                  const ap::integer_1d_array& nbd, int& task, int& info,
                  int& k) {
  int i;

  if (n <= 0) {
    task = 2;
  }
  if (m <= 0) {
    task = 2;
  }
  if (m > n) {
    task = 2;
  }
  if (factr < 0) {
    task = 2;
  }
  for (i = 1; i <= n; i++) {
    if (nbd(i) < 0 || nbd(i) > 3) {
      task = 2;
      info = -6;
      k = i;
    }
    if (nbd(i) == 2) {
      if (l(i) > u(i)) {
        task = 2;
        info = -7;
        k = i;
      }
    }
  }
}

template <typename real>
void lbfgsbformk(const int& n, const int& nsub, const ap::integer_1d_array& ind,
                 const int& nenter, const int& ileave,
                 const ap::integer_1d_array& indx2, const int& iupdat,
                 const bool& updatd, ap::template_2d_array<real, true>& wn,
                 ap::template_2d_array<real, true>& wn1, const int& m,
                 const ap::template_2d_array<real, true>& ws,
                 const ap::template_2d_array<real, true>& wy,
                 const ap::template_2d_array<real, true>& sy, const real& theta,
                 const int& col, const int& head, int& info,
                 ap::template_1d_array<real, true>& workvec,
                 ap::template_2d_array<real, true>& workmat) {
  int m2;
  int ipntr;
  int jpntr;
  int iy;
  int iis;
  int jy;
  int js;
  int is1;
  int js1;
  int k1;
  int i;
  int k;
  int col2;
  int pbegin;
  int pend;
  int dbegin;
  int dend;
  int upcl;
  real temp1;
  real temp2;
  real temp3;
  real temp4;
  real v;
  int j;

  if (updatd) {
    if (iupdat > m) {
      for (jy = 1; jy <= m - 1; jy++) {
        js = m + jy;
        ap::vmove<real>(wn1.getcolumn(jy, jy, m - 1),
                        wn1.getcolumn(jy + 1, jy + 1, m));
        ap::vmove<real>(wn1.getcolumn(js, js, js + m - jy - 1),
                        wn1.getcolumn(js + 1, js + 1, js + m - jy));
        ap::vmove<real>(wn1.getcolumn(jy, m + 1, m + m - 1),
                        wn1.getcolumn(jy + 1, m + 2, m + m));
      }
    }
    pbegin = 1;
    pend = nsub;
    dbegin = nsub + 1;
    dend = n;
    iy = col;
    iis = m + col;
    ipntr = head + col - 1;
    if (ipntr > m) {
      ipntr = ipntr - m;
    }
    jpntr = head;
    for (jy = 1; jy <= col; jy++) {
      js = m + jy;
      temp1 = 0;
      temp2 = 0;
      temp3 = 0;
      for (k = pbegin; k <= pend; k++) {
        k1 = ind(k);
        temp1 = temp1 + wy(k1, ipntr) * wy(k1, jpntr);
      }
      for (k = dbegin; k <= dend; k++) {
        k1 = ind(k);
        temp2 = temp2 + ws(k1, ipntr) * ws(k1, jpntr);
        temp3 = temp3 + ws(k1, ipntr) * wy(k1, jpntr);
      }
      wn1(iy, jy) = temp1;
      wn1(iis, js) = temp2;
      wn1(iis, jy) = temp3;
      jpntr = jpntr % m + 1;
    }
    jy = col;
    jpntr = head + col - 1;
    if (jpntr > m) {
      jpntr = jpntr - m;
    }
    ipntr = head;
    for (i = 1; i <= col; i++) {
      iis = m + i;
      temp3 = 0;
      for (k = pbegin; k <= pend; k++) {
        k1 = ind(k);
        temp3 = temp3 + ws(k1, ipntr) * wy(k1, jpntr);
      }
      ipntr = ipntr % m + 1;
      wn1(iis, jy) = temp3;
    }
    upcl = col - 1;
  } else {
    upcl = col;
  }
  ipntr = head;
  for (iy = 1; iy <= upcl; iy++) {
    iis = m + iy;
    jpntr = head;
    for (jy = 1; jy <= iy; jy++) {
      js = m + jy;
      temp1 = 0;
      temp2 = 0;
      temp3 = 0;
      temp4 = 0;
      for (k = 1; k <= nenter; k++) {
        k1 = indx2(k);
        temp1 = temp1 + wy(k1, ipntr) * wy(k1, jpntr);
        temp2 = temp2 + ws(k1, ipntr) * ws(k1, jpntr);
      }
      for (k = ileave; k <= n; k++) {
        k1 = indx2(k);
        temp3 = temp3 + wy(k1, ipntr) * wy(k1, jpntr);
        temp4 = temp4 + ws(k1, ipntr) * ws(k1, jpntr);
      }
      wn1(iy, jy) = wn1(iy, jy) + temp1 - temp3;
      wn1(iis, js) = wn1(iis, js) - temp2 + temp4;
      jpntr = jpntr % m + 1;
    }
    ipntr = ipntr % m + 1;
  }
  ipntr = head;
  for (iis = m + 1; iis <= m + upcl; iis++) {
    jpntr = head;
    for (jy = 1; jy <= upcl; jy++) {
      temp1 = 0;
      temp3 = 0;
      for (k = 1; k <= nenter; k++) {
        k1 = indx2(k);
        temp1 = temp1 + ws(k1, ipntr) * wy(k1, jpntr);
      }
      for (k = ileave; k <= n; k++) {
        k1 = indx2(k);
        temp3 = temp3 + ws(k1, ipntr) * wy(k1, jpntr);
      }
      if (iis <= jy + m) {
        wn1(iis, jy) = wn1(iis, jy) + temp1 - temp3;
      } else {
        wn1(iis, jy) = wn1(iis, jy) - temp1 + temp3;
      }
      jpntr = jpntr % m + 1;
    }
    ipntr = ipntr % m + 1;
  }
  m2 = 2 * m;
  for (iy = 1; iy <= col; iy++) {
    iis = col + iy;
    is1 = m + iy;
    for (jy = 1; jy <= iy; jy++) {
      js = col + jy;
      js1 = m + jy;
      wn(jy, iy) = wn1(iy, jy) / theta;
      wn(js, iis) = wn1(is1, js1) * theta;
    }
    for (jy = 1; jy <= iy - 1; jy++) {
      wn(jy, iis) = -wn1(is1, jy);
    }
    for (jy = iy; jy <= col; jy++) {
      wn(jy, iis) = wn1(is1, jy);
    }
    wn(iy, iy) = wn(iy, iy) + sy(iy, iy);
  }
  info = 0;
  if (!lbfgsbdpofa<real>(wn, col)) {
    info = -1;
    return;
  }
  col2 = 2 * col;
  for (js = col + 1; js <= col2; js++) {
    ap::vmove<real>(workvec.getvector(1, col), wn.getcolumn(js, 1, col));
    lbfgsbdtrsl<real>(wn, col, workvec, 11, info);
    ap::vmove<real>(wn.getcolumn(js, 1, col), workvec.getvector(1, col));
  }
  for (iis = col + 1; iis <= col2; iis++) {
    for (js = iis; js <= col2; js++) {
      v = ap::vdotproduct<real>(wn.getcolumn(iis, 1, col),
                                wn.getcolumn(js, 1, col));
      wn(iis, js) = wn(iis, js) + v;
    }
  }
  for (j = 1; j <= col; j++) {
    ap::vmove<real>(&workmat(j, 1), &wn(col + j, col + 1), ap::vlen(1, col));
  }
  info = 0;
  if (!lbfgsbdpofa<real>(workmat, col)) {
    info = -2;
    return;
  }
  for (j = 1; j <= col; j++) {
    ap::vmove<real>(&wn(col + j, col + 1), &workmat(j, 1),
                    ap::vlen(col + 1, col + col));
  }
}

template <typename real>
void lbfgsbformt(const int& m, ap::template_2d_array<real, true>& wt,
                 const ap::template_2d_array<real, true>& sy,
                 const ap::template_2d_array<real, true>& ss, const int& col,
                 const real& theta, int& info) {
  int i;
  int j;
  int k;
  int k1;
  real ddum;

  for (j = 1; j <= col; j++) {
    wt(1, j) = theta * ss(1, j);
  }
  for (i = 2; i <= col; i++) {
    for (j = i; j <= col; j++) {
      k1 = ap::minint(i, j) - 1;
      ddum = 0;
      for (k = 1; k <= k1; k++) {
        ddum = ddum + sy(i, k) * sy(j, k) / sy(k, k);
      }
      wt(i, j) = ddum + theta * ss(i, j);
    }
  }
  info = 0;
  if (!lbfgsbdpofa<real>(wt, col)) {
    info = -3;
  }
}

void lbfgsbfreev(const int& n, int& nfree, ap::integer_1d_array& index,
                 int& nenter, int& ileave, ap::integer_1d_array& indx2,
                 const ap::integer_1d_array& iwhere, bool& wrk,
                 const bool& updatd, const bool& cnstnd, const int& iter) {
  int iact;
  int i;
  int k;

  nenter = 0;
  ileave = n + 1;
  if (iter > 0 && cnstnd) {
    for (i = 1; i <= nfree; i++) {
      k = index(i);
      if (iwhere(k) > 0) {
        ileave = ileave - 1;
        indx2(ileave) = k;
      }
    }
    for (i = 1 + nfree; i <= n; i++) {
      k = index(i);
      if (iwhere(k) <= 0) {
        nenter = nenter + 1;
        indx2(nenter) = k;
      }
    }
  }
  wrk = ileave < n + 1 || nenter > 0 || updatd;
  nfree = 0;
  iact = n + 1;
  for (i = 1; i <= n; i++) {
    if (iwhere(i) <= 0) {
      nfree = nfree + 1;
      index(nfree) = i;
    } else {
      iact = iact - 1;
      index(iact) = i;
    }
  }
}

template <typename real>
void lbfgsbhpsolb(const int& n, ap::template_1d_array<real, true>& t,
                  ap::integer_1d_array& iorder, const int& iheap) {
  int i;
  int j;
  int k;
  int indxin;
  int indxou;
  real ddum;
  real dout;

  if (iheap == 0) {
    for (k = 2; k <= n; k++) {
      ddum = t(k);
      indxin = iorder(k);
      i = k;
      while (true) {
        if (i > 1) {
          j = i / 2;
          if (ddum < t(j)) {
            t(i) = t(j);
            iorder(i) = iorder(j);
            i = j;
            continue;
          }
        }
        break;
      }
      t(i) = ddum;
      iorder(i) = indxin;
    }
  }
  if (n > 1) {
    i = 1;
    dout = t(1);
    indxou = iorder(1);
    ddum = t(n);
    indxin = iorder(n);
    while (true) {
      j = i + i;
      if (j <= n - 1) {
        if (t(j + 1) < t(j)) {
          j = j + 1;
        }
        if (t(j) < ddum) {
          t(i) = t(j);
          iorder(i) = iorder(j);
          i = j;
          continue;
        }
      }
      break;
    }
    t(i) = ddum;
    iorder(i) = indxin;
    t(n) = dout;
    iorder(n) = indxou;
  }
}

template <typename real>
void lbfgsblnsrlb(
    const int& n, const ap::template_1d_array<real, true>& l,
    const ap::template_1d_array<real, true>& u, const ap::integer_1d_array& nbd,
    ap::template_1d_array<real, true>& x, const real& f, real& fold, real& gd,
    real& gdold, const ap::template_1d_array<real, true>& g,
    const ap::template_1d_array<real, true>& d,
    ap::template_1d_array<real, true>& r, ap::template_1d_array<real, true>& t,
    const ap::template_1d_array<real, true>& z, real& stp, real& dnrm,
    real& dtd, real& xstep, real& stpmx, const int& iter, int& ifun, int& iback,
    int& nfgv, int& info, int& task, const bool& boxed, const bool& cnstnd,
    int& csave, ap::integer_1d_array& isave,
    ap::template_1d_array<real, true>& dsave) {
  int i;
  real a1;
  real a2;
  real v;
  real ftol;
  real gtol;
  real xtol;
  real big;
  int addinfo;

  addinfo = 0;
  big = static_cast<real>(1.0E10);
  ftol = static_cast<real>(1.0E-3);
  gtol = static_cast<real>(0.9E0);
  xtol = static_cast<real>(0.1E0);
  if (task != 1) {
    v = ap::vdotproduct<real>(&d(1), &d(1), ap::vlen(1, n));
    dtd = v;
    dnrm = sqrt(dtd);
    stpmx = big;
    if (cnstnd) {
      if (iter == 0) {
        stpmx = 1;
      } else {
        for (i = 1; i <= n; i++) {
          a1 = d(i);
          if (nbd(i) != 0) {
            if (a1 < 0 && nbd(i) <= 2) {
              a2 = l(i) - x(i);
              if (a2 >= 0) {
                stpmx = 0;
              } else {
                if (a1 * stpmx < a2) {
                  stpmx = a2 / a1;
                }
              }
            } else {
              if (a1 > 0 && nbd(i) >= 2) {
                a2 = u(i) - x(i);
                if (a2 <= 0) {
                  stpmx = 0;
                } else {
                  if (a1 * stpmx > a2) {
                    stpmx = a2 / a1;
                  }
                }
              }
            }
          }
        }
      }
    }
    if (iter == 0 && !boxed) {
      stp = ap::minreal<real>(1 / dnrm, stpmx);
    } else {
      stp = 1;
    }
    ap::vmove<real>(&t(1), &x(1), ap::vlen(1, n));
    ap::vmove<real>(&r(1), &g(1), ap::vlen(1, n));
    fold = f;
    ifun = 0;
    iback = 0;
    csave = 0;
  }
  v = ap::vdotproduct<real>(&g(1), &d(1), ap::vlen(1, n));
  gd = v;
  if (ifun == 0) {
    gdold = gd;
    if (gd >= 0) {
      info = -4;
      return;
    }
  }
  lbfgsbdcsrch<real>(f, gd, stp, ftol, gtol, xtol, real(0), stpmx, csave, isave,
                     dsave, addinfo);
  xstep = stp * dnrm;
  if (csave != 4 && csave != 3) {
    task = 1;
    ifun = ifun + 1;
    nfgv = nfgv + 1;
    iback = ifun - 1;
    if (stp == 1) {
      ap::vmove<real>(&x(1), &z(1), ap::vlen(1, n));
    } else {
      for (i = 1; i <= n; i++) {
        x(i) = stp * d(i) + t(i);
      }
    }
  } else {
    task = 5;
  }
}

template <typename real>
void lbfgsbmatupd(const int& n, const int& m,
                  ap::template_2d_array<real, true>& ws,
                  ap::template_2d_array<real, true>& wy,
                  ap::template_2d_array<real, true>& sy,
                  ap::template_2d_array<real, true>& ss,
                  const ap::template_1d_array<real, true>& d,
                  const ap::template_1d_array<real, true>& r, int& itail,
                  const int& iupdat, int& col, int& head, real& theta,
                  const real& rr, const real& dr, const real& stp,
                  const real& dtd) {
  int j;
  int pointr;
  real v;

  if (iupdat <= m) {
    col = iupdat;
    itail = (head + iupdat - 2) % m + 1;
  } else {
    itail = itail % m + 1;
    head = head % m + 1;
  }
  ap::vmove<real>(ws.getcolumn(itail, 1, n), d.getvector(1, n));
  ap::vmove<real>(wy.getcolumn(itail, 1, n), r.getvector(1, n));
  theta = rr / dr;
  if (iupdat > m) {
    for (j = 1; j <= col - 1; j++) {
      ap::vmove<real>(ss.getcolumn(j, 1, j), ss.getcolumn(j + 1, 2, j + 1));
      ap::vmove<real>(sy.getcolumn(j, j, col - 1),
                      sy.getcolumn(j + 1, j + 1, col));
    }
  }
  pointr = head;
  for (j = 1; j <= col - 1; j++) {
    v = ap::vdotproduct<real>(d.getvector(1, n), wy.getcolumn(pointr, 1, n));
    sy(col, j) = v;
    v = ap::vdotproduct<real>(ws.getcolumn(pointr, 1, n), d.getvector(1, n));
    ss(j, col) = v;
    pointr = pointr % m + 1;
  }
  if (stp == 1) {
    ss(col, col) = dtd;
  } else {
    ss(col, col) = stp * stp * dtd;
  }
  sy(col, col) = dr;
}

template <typename real>
void lbfgsbprojgr(const int& n, const ap::template_1d_array<real, true>& l,
                  const ap::template_1d_array<real, true>& u,
                  const ap::integer_1d_array& nbd,
                  const ap::template_1d_array<real, true>& x,
                  const ap::template_1d_array<real, true>& g, real& sbgnrm) {
  int i;
  real gi;

  sbgnrm = 0;
  for (i = 1; i <= n; i++) {
    gi = g(i);
    if (nbd(i) != 0) {
      if (gi < 0) {
        if (nbd(i) >= 2) {
          gi = ap::maxreal<real>(x(i) - u(i), gi);
        }
      } else {
        if (nbd(i) <= 2) {
          gi = ap::minreal<real>(x(i) - l(i), gi);
        }
      }
    }
    sbgnrm = ap::maxreal<real>(sbgnrm, static_cast<real>(fabs(gi)));
  }
}

template <typename real>
void lbfgsbsubsm(const int& n, const int& m, const int& nsub,
                 const ap::integer_1d_array& ind,
                 const ap::template_1d_array<real, true>& l,
                 const ap::template_1d_array<real, true>& u,
                 const ap::integer_1d_array& nbd,
                 ap::template_1d_array<real, true>& x,
                 ap::template_1d_array<real, true>& d,
                 const ap::template_2d_array<real, true>& ws,
                 const ap::template_2d_array<real, true>& wy, const real& theta,
                 const int& col, const int& head, int& iword,
                 ap::template_1d_array<real, true>& wv,
                 ap::template_2d_array<real, true>& wn, int& info) {
  int pointr;
  int m2;
  int col2;
  int ibd;
  int jy;
  int js;
  int i;
  int j;
  int k;
  real alpha;
  real dk;
  real temp1;
  real temp2;

  if (nsub <= 0) {
    return;
  }
  pointr = head;
  for (i = 1; i <= col; i++) {
    temp1 = 0;
    temp2 = 0;
    for (j = 1; j <= nsub; j++) {
      k = ind(j);
      temp1 = temp1 + wy(k, pointr) * d(j);
      temp2 = temp2 + ws(k, pointr) * d(j);
    }
    wv(i) = temp1;
    wv(col + i) = theta * temp2;
    pointr = pointr % m + 1;
  }
  m2 = 2 * m;
  col2 = 2 * col;
  lbfgsbdtrsl<real>(wn, col2, wv, 11, info);
  if (info != 0) {
    return;
  }
  for (i = 1; i <= col; i++) {
    wv(i) = -wv(i);
  }
  lbfgsbdtrsl<real>(wn, col2, wv, 1, info);
  if (info != 0) {
    return;
  }
  pointr = head;
  for (jy = 1; jy <= col; jy++) {
    js = col + jy;
    for (i = 1; i <= nsub; i++) {
      k = ind(i);
      d(i) = d(i) + wy(k, pointr) * wv(jy) / theta + ws(k, pointr) * wv(js);
    }
    pointr = pointr % m + 1;
  }
  for (i = 1; i <= nsub; i++) {
    d(i) = d(i) / theta;
  }
  alpha = 1;
  temp1 = alpha;
  for (i = 1; i <= nsub; i++) {
    k = ind(i);
    dk = d(i);
    if (nbd(k) != 0) {
      if (dk < 0 && nbd(k) <= 2) {
        temp2 = l(k) - x(k);
        if (temp2 >= 0) {
          temp1 = 0;
        } else {
          if (dk * alpha < temp2) {
            temp1 = temp2 / dk;
          }
        }
      } else {
        if (dk > 0 && nbd(k) >= 2) {
          temp2 = u(k) - x(k);
          if (temp2 <= 0) {
            temp1 = 0;
          } else {
            if (dk * alpha > temp2) {
              temp1 = temp2 / dk;
            }
          }
        }
      }
      if (temp1 < alpha) {
        alpha = temp1;
        ibd = i;
      }
    }
  }
  if (alpha < 1) {
    dk = d(ibd);
    k = ind(ibd);
    if (dk > 0) {
      x(k) = u(k);
      d(ibd) = 0;
    } else {
      if (dk < 0) {
        x(k) = l(k);
        d(ibd) = 0;
      }
    }
  }
  for (i = 1; i <= nsub; i++) {
    k = ind(i);
    x(k) = x(k) + alpha * d(i);
  }
  if (alpha < 1) {
    iword = 1;
  } else {
    iword = 0;
  }
}

template <typename real>
void lbfgsbdcsrch(const real& f, const real& g, real& stp, const real& ftol,
                  const real& gtol, const real& xtol, const real& stpmin,
                  const real& stpmax, int& task, ap::integer_1d_array& isave,
                  ap::template_1d_array<real, true>& dsave, int& addinfo) {
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
  real xtrapl;
  real xtrapu;

  xtrapl = static_cast<real>(1.1E0);
  xtrapu = static_cast<real>(4.0E0);
  while (true) {
    if (task == 0) {
      if (stp < stpmin) {
        task = 2;
        addinfo = 0;
      }
      if (stp > stpmax) {
        task = 2;
        addinfo = 0;
      }
      if (g >= 0) {
        task = 2;
        addinfo = 0;
      }
      if (ftol < 0) {
        task = 2;
        addinfo = 0;
      }
      if (gtol < 0) {
        task = 2;
        addinfo = 0;
      }
      if (xtol < 0) {
        task = 2;
        addinfo = 0;
      }
      if (stpmin < 0) {
        task = 2;
        addinfo = 0;
      }
      if (stpmax < stpmin) {
        task = 2;
        addinfo = 0;
      }
      if (task == 2) {
        return;
      }
      brackt = false;
      stage = 1;
      finit = f;
      ginit = g;
      gtest = ftol * ginit;
      width = stpmax - stpmin;
      width1 = width / static_cast<real>(0.5);
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
      if (isave(1) == 1) {
        brackt = true;
      } else {
        brackt = false;
      }
      stage = isave(2);
      ginit = dsave(1);
      gtest = dsave(2);
      gx = dsave(3);
      gy = dsave(4);
      finit = dsave(5);
      fx = dsave(6);
      fy = dsave(7);
      stx = dsave(8);
      sty = dsave(9);
      stmin = dsave(10);
      stmax = dsave(11);
      width = dsave(12);
      width1 = dsave(13);
    }
    ftest = finit + stp * gtest;
    if (stage == 1 && f <= ftest && g >= 0) {
      stage = 2;
    }
    if (brackt && (stp <= stmin || stp >= stmax)) {
      task = 3;
      addinfo = 1;
    }
    if (brackt && stmax - stmin <= xtol * stmax) {
      task = 3;
      addinfo = 2;
    }
    if (stp == stpmax && f <= ftest && g <= gtest) {
      task = 3;
      addinfo = 3;
    }
    if (stp == stpmin && (f > ftest || g >= gtest)) {
      task = 3;
      addinfo = 4;
    }
    if (f <= ftest && fabs(g) <= gtol * (-ginit)) {
      task = 4;
      addinfo = -1;
    }
    if (task == 3 || task == 4) {
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
      if (static_cast<real>(fabs(sty - stx)) >=
          static_cast<real>(0.66) * width1) {
        stp = stx + static_cast<real>(0.5) * (sty - stx);
      }
      width1 = width;
      width = static_cast<real>(fabs(sty - stx));
    }
    if (brackt) {
      stmin = ap::minreal<real>(stx, sty);
      stmax = ap::maxreal<real>(stx, sty);
    } else {
      stmin = stp + xtrapl * (stp - stx);
      stmax = stp + xtrapu * (stp - stx);
    }
    stp = ap::maxreal<real>(stp, stpmin);
    stp = ap::minreal<real>(stp, stpmax);
    if (brackt && (stp <= stmin || stp >= stmax) ||
        brackt && stmax - stmin <= xtol * stmax) {
      stp = stx;
    }
    task = 1;
    break;
  }
  if (brackt) {
    isave(1) = 1;
  } else {
    isave(1) = 0;
  }
  isave(2) = stage;
  dsave(1) = ginit;
  dsave(2) = gtest;
  dsave(3) = gx;
  dsave(4) = gy;
  dsave(5) = finit;
  dsave(6) = fx;
  dsave(7) = fy;
  dsave(8) = stx;
  dsave(9) = sty;
  dsave(10) = stmin;
  dsave(11) = stmax;
  dsave(12) = width;
  dsave(13) = width1;
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

  sgnd = dp * (dx / static_cast<real>(fabs(dx)));
  if (fp > fx) {
    theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
    s = ap::maxreal<real>(static_cast<real>(fabs(theta)),
                          ap::maxreal<real>(static_cast<real>(fabs(dx)),
                                            static_cast<real>(fabs(dp))));
    gamma = s * sqrt(ap::sqr(theta / s) - dx / s * (dp / s));
    if (stp < stx) {
      gamma = -gamma;
    }
    p = gamma - dx + theta;
    q = gamma - dx + gamma + dp;
    r = p / q;
    stpc = stx + r * (stp - stx);
    stpq = stx + dx / ((fx - fp) / (stp - stx) + dx) / 2 * (stp - stx);
    if (fabs(stpc - stx) < fabs(stpq - stx)) {
      stpf = stpc;
    } else {
      stpf = stpc + (stpq - stpc) / 2;
    }
    brackt = true;
  } else {
    if (sgnd < 0) {
      theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      s = ap::maxreal<real>(fabs(theta), ap::maxreal<real>(fabs(dx), fabs(dp)));
      gamma = s * sqrt(ap::sqr(theta / s) - dx / s * (dp / s));
      if (stp > stx) {
        gamma = -gamma;
      }
      p = gamma - dp + theta;
      q = gamma - dp + gamma + dx;
      r = p / q;
      stpc = stp + r * (stx - stp);
      stpq = stp + dp / (dp - dx) * (stx - stp);
      if (fabs(stpc - stp) > fabs(stpq - stp)) {
        stpf = stpc;
      } else {
        stpf = stpq;
      }
      brackt = true;
    } else {
      if (fabs(dp) < fabs(dx)) {
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
        s = ap::maxreal<real>(fabs(theta),
                              ap::maxreal<real>(fabs(dx), fabs(dp)));
        gamma = s * sqrt(ap::maxreal<real>(
                        real(0), ap::sqr(theta / s) - dx / s * (dp / s)));
        if (stp > stx) {
          gamma = -gamma;
        }
        p = gamma - dp + theta;
        q = gamma + (dx - dp) + gamma;
        r = p / q;
        if (r < 0 && gamma != 0) {
          stpc = stp + r * (stx - stp);
        } else {
          if (stp > stx) {
            stpc = stpmax;
          } else {
            stpc = stpmin;
          }
        }
        stpq = stp + dp / (dp - dx) * (stx - stp);
        if (brackt) {
          if (fabs(stpc - stp) < fabs(stpq - stp)) {
            stpf = stpc;
          } else {
            stpf = stpq;
          }
          if (stp > stx) {
            stpf = ap::minreal<real>(stp + 0.66 * (sty - stp), stpf);
          } else {
            stpf = ap::maxreal<real>(stp + 0.66 * (sty - stp), stpf);
          }
        } else {
          if (fabs(stpc - stp) > fabs(stpq - stp)) {
            stpf = stpc;
          } else {
            stpf = stpq;
          }
          stpf = ap::minreal<real>(stpmax, stpf);
          stpf = ap::maxreal<real>(stpmin, stpf);
        }
      } else {
        if (brackt) {
          theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
          s = ap::maxreal<real>(fabs(theta),
                                ap::maxreal<real>(fabs(dy), fabs(dp)));
          gamma = s * sqrt(ap::sqr(theta / s) - dy / s * (dp / s));
          if (stp > sty) {
            gamma = -gamma;
          }
          p = gamma - dp + theta;
          q = gamma - dp + gamma + dy;
          r = p / q;
          stpc = stp + r * (sty - stp);
          stpf = stpc;
        } else {
          if (stp > stx) {
            stpf = stpmax;
          } else {
            stpf = stpmin;
          }
        }
      }
    }
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
bool lbfgsbdpofa(ap::template_2d_array<real, true>& a, const int& n) {
  bool result;
  real t;
  real s;
  real v;
  int j;
  int jm1;
  int k;

  for (j = 1; j <= n; j++) {
    s = 0.0;
    jm1 = j - 1;
    if (jm1 >= 1) {
      for (k = 1; k <= jm1; k++) {
        v = ap::vdotproduct<real>(a.getcolumn(k, 1, k - 1),
                                  a.getcolumn(j, 1, k - 1));
        t = a(k, j) - v;
        t = t / a(k, k);
        a(k, j) = t;
        s = s + t * t;
      }
    }
    s = a(j, j) - s;
    if (s <= 0.0) {
      result = false;
      return result;
    }
    a(j, j) = sqrt(s);
  }
  result = true;
  return result;
}

template <typename real>
void lbfgsbdtrsl(ap::template_2d_array<real, true>& t, const int& n,
                 ap::template_1d_array<real, true>& b, const int& job,
                 int& info) {
  real temp;
  real v;
  int cse;
  int j;
  int jj;

  for (j = 1; j <= n; j++) {
    if (t(j, j) == 0.0) {
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
    b(1) = b(1) / t(1, 1);
    if (n < 2) {
      return;
    }
    for (j = 2; j <= n; j++) {
      temp = -b(j - 1);
      ap::vadd<real>(b.getvector(j, n), t.getcolumn(j - 1, j, n), temp);
      b(j) = b(j) / t(j, j);
    }
    return;
  }
  if (cse == 2) {
    b(n) = b(n) / t(n, n);
    if (n < 2) {
      return;
    }
    for (jj = 2; jj <= n; jj++) {
      j = n - jj + 1;
      temp = -b(j + 1);
      ap::vadd<real>(b.getvector(1, j), t.getcolumn(j + 1, 1, j), temp);
      b(j) = b(j) / t(j, j);
    }
    return;
  }
  if (cse == 3) {
    b(n) = b(n) / t(n, n);
    if (n < 2) {
      return;
    }
    for (jj = 2; jj <= n; jj++) {
      j = n - jj + 1;
      v = ap::vdotproduct<real>(t.getcolumn(j, j + 1, j + 1 + jj - 1 - 1),
                                b.getvector(j + 1, j + 1 + jj - 1 - 1));
      b(j) = b(j) - v;
      b(j) = b(j) / t(j, j);
    }
    return;
  }
  if (cse == 4) {
    b(1) = b(1) / t(1, 1);
    if (n < 2) {
      return;
    }
    for (j = 2; j <= n; j++) {
      v = ap::vdotproduct<real>(t.getcolumn(j, 1, j - 1),
                                b.getvector(1, j - 1));
      b(j) = b(j) - v;
      b(j) = b(j) / t(j, j);
    }
    return;
  }
}

#define INST_HELPER(real)                                                      \
  template void lbfgsbactive<real>(                                            \
      const int&, const ap::template_1d_array<real, true>&,                    \
      const ap::template_1d_array<real, true>&, const ap::integer_1d_array&,   \
      ap::template_1d_array<real, true>&, ap::integer_1d_array&, bool&, bool&, \
      bool&);                                                                  \
  template void lbfgsbbmv<real>(                                               \
      const int&, const ap::template_2d_array<real, true>&,                    \
      ap::template_2d_array<real, true>&, const int&,                          \
      const ap::template_1d_array<real, true>&,                                \
      ap::template_1d_array<real, true>&, int&,                                \
      ap::template_1d_array<real, true>&);                                     \
  template void lbfgsbcauchy<real>(                                            \
      const int&, const ap::template_1d_array<real, true>&,                    \
      const ap::template_1d_array<real, true>&,                                \
      const ap::template_1d_array<real, true>&, const ap::integer_1d_array&,   \
      const ap::template_1d_array<real, true>&, ap::integer_1d_array&,         \
      ap::integer_1d_array&, ap::template_1d_array<real, true>&,               \
      ap::template_1d_array<real, true>&, ap::template_1d_array<real, true>&,  \
      const int&, const ap::template_2d_array<real, true>&,                    \
      const ap::template_2d_array<real, true>&,                                \
      const ap::template_2d_array<real, true>&,                                \
      ap::template_2d_array<real, true>&, const real&, const int&, const int&, \
      ap::template_1d_array<real, true>&, ap::template_1d_array<real, true>&,  \
      ap::template_1d_array<real, true>&, ap::template_1d_array<real, true>&,  \
      int&, const ap::template_1d_array<real, true>&,                          \
      const ap::template_1d_array<real, true>&, const real&, int&,             \
      ap::template_1d_array<real, true>&);                                     \
  template void lbfgsbcmprlb<real>(                                            \
      const int&, const int&, const ap::template_1d_array<real, true>&,        \
      const ap::template_1d_array<real, true>&,                                \
      const ap::template_2d_array<real, true>&,                                \
      const ap::template_2d_array<real, true>&,                                \
      const ap::template_2d_array<real, true>&,                                \
      ap::template_2d_array<real, true>&,                                      \
      const ap::template_1d_array<real, true>&,                                \
      ap::template_1d_array<real, true>&, ap::template_1d_array<real, true>&,  \
      const ap::integer_1d_array&, const real&, const int& col, const int&,    \
      const int&, const bool&, int&, ap::template_1d_array<real, true>&,       \
      ap::template_1d_array<real, true>&);                                     \
  template void lbfgsberrclb<real>(const int&, const int&, const real&,        \
                                   const ap::template_1d_array<real, true>&,   \
                                   const ap::template_1d_array<real, true>&,   \
                                   const ap::integer_1d_array&, int&, int&,    \
                                   int&);                                      \
  template void lbfgsbformk<real>(                                             \
      const int&, const int&, const ap::integer_1d_array&, const int&,         \
      const int&, const ap::integer_1d_array&, const int&, const bool&,        \
      ap::template_2d_array<real, true>&, ap::template_2d_array<real, true>&,  \
      const int&, const ap::template_2d_array<real, true>&,                    \
      const ap::template_2d_array<real, true>&,                                \
      const ap::template_2d_array<real, true>&, const real&, const int&,       \
      const int&, int&, ap::template_1d_array<real, true>&,                    \
      ap::template_2d_array<real, true>&);                                     \
  template void lbfgsbformt<real>(const int&,                                  \
                                  ap::template_2d_array<real, true>&,          \
                                  const ap::template_2d_array<real, true>&,    \
                                  const ap::template_2d_array<real, true>&,    \
                                  const int&, const real&, int&);              \
  template void lbfgsbhpsolb<real>(const int&,                                 \
                                   ap::template_1d_array<real, true>&,         \
                                   ap::integer_1d_array&, const int&);         \
  template void lbfgsblnsrlb<real>(                                            \
      const int&, const ap::template_1d_array<real, true>&,                    \
      const ap::template_1d_array<real, true>&, const ap::integer_1d_array&,   \
      ap::template_1d_array<real, true>&, const real&, real&, real&, real&,    \
      const ap::template_1d_array<real, true>&,                                \
      const ap::template_1d_array<real, true>&,                                \
      ap::template_1d_array<real, true>&, ap::template_1d_array<real, true>&,  \
      const ap::template_1d_array<real, true>&, real&, real&, real&, real&,    \
      real&, const int&, int&, int&, int&, int&, int&, const bool&,            \
      const bool&, int&, ap::integer_1d_array&,                                \
      ap::template_1d_array<real, true>&);                                     \
  template void lbfgsbmatupd<real>(                                            \
      const int&, const int&, ap::template_2d_array<real, true>&,              \
      ap::template_2d_array<real, true>&, ap::template_2d_array<real, true>&,  \
      ap::template_2d_array<real, true>&,                                      \
      const ap::template_1d_array<real, true>&,                                \
      const ap::template_1d_array<real, true>&, int&, const int&, int&, int&,  \
      real&, const real&, const real&, const real&, const real&);              \
  template void lbfgsbprojgr<real>(                                            \
      const int&, const ap::template_1d_array<real, true>&,                    \
      const ap::template_1d_array<real, true>&, const ap::integer_1d_array&,   \
      const ap::template_1d_array<real, true>&,                                \
      const ap::template_1d_array<real, true>&, real&);                        \
  template void lbfgsbsubsm<real>(                                             \
      const int&, const int&, const int&, const ap::integer_1d_array&,         \
      const ap::template_1d_array<real, true>&,                                \
      const ap::template_1d_array<real, true>&, const ap::integer_1d_array&,   \
      ap::template_1d_array<real, true>&, ap::template_1d_array<real, true>&,  \
      const ap::template_2d_array<real, true>&,                                \
      const ap::template_2d_array<real, true>&, const real&, const int&,       \
      const int&, int&, ap::template_1d_array<real, true>&,                    \
      ap::template_2d_array<real, true>&, int&);                               \
  template void lbfgsbdcsrch<real>(                                            \
      const real&, const real&, real&, const real&, const real&, const real&,  \
      const real&, const real&, int& task, ap::integer_1d_array&,              \
      ap::template_1d_array<real, true>&, int&);                               \
  template void lbfgsbdcstep<real>(real&, real&, real&, real&, real&, real&,   \
                                   real&, const real&, const real&, bool&,     \
                                   const real&, const real&);                  \
  template bool lbfgsbdpofa<real>(ap::template_2d_array<real, true>&,          \
                                  const int&);                                 \
  template void lbfgsbdtrsl<real>(                                             \
      ap::template_2d_array<real, true>&, const int&,                          \
      ap::template_1d_array<real, true>&, const int&, int&);                   \
  template void lbfgsbminimize<real>(                                          \
      const int&, const LBFGSB_CUDA_STATE<real>&,                              \
      const LBFGSB_CUDA_OPTION<real>&, real*, const int*, const real*,         \
      const real*, LBFGSB_CUDA_SUMMARY<real>&);

INST_HELPER(double);
INST_HELPER(float);

}  // namespace cpu
}  // namespace lbfgsbcuda

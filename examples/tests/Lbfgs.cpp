#include "Lbfgs.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <sstream>

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

class InternalError: public std::exception {
private:
  std::string m_message;
public:
  InternalError(): m_message("none") {}
  virtual ~InternalError() = default;
  InternalError(std::string message) throw(): m_message(message) { }
  InternalError(const char *message) throw(): m_message(message) { }
  virtual const char *what() const throw() { return m_message.c_str(); }
};

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err) {
    std::stringstream ss;
    ss << "Cuda Error in " << file << " @ " << line << ". Error code: " << err << " Error string: " << cudaGetErrorString(err);
    std::cerr << ss.str() << "\n";
    throw InternalError(ss.str());
  }
}

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__);
#endif

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <typename T>
LBFGS<T>::LBFGS(int dim, int historySize) : m_dim(dim), m_historySize(historySize)
{
  memset(&m_state, 0, sizeof(m_state));
  m_state.m_cublas_handle = 0;
  memset(&m_lbfgsb_options, 0, sizeof(m_lbfgsb_options));
  memset(&m_summary, 0, sizeof(m_summary));
  m_deviceBuffersAllocated = false;
}

template <typename T>
LBFGS<T>::~LBFGS()
{
  freeDeviceBuffers();
}

template <typename T>
void LBFGS<T>::allocateDeviceBuffers()
{
  if (m_deviceBuffersAllocated) {
    return;
  }
  checkCudaErrors(cudaMalloc((void**)&m_d_xu,m_dim*sizeof(T)));
  checkCudaErrors(cudaMalloc((void**)&m_d_xl,m_dim*sizeof(T)));
  checkCudaErrors(cudaMalloc((void**)&m_d_nbd,m_dim*sizeof(int)));
  m_deviceBuffersAllocated = true;
}

template <typename T>
void LBFGS<T>::freeDeviceBuffers()
{
  if (not m_deviceBuffersAllocated) {
    return;
  }
  if (m_d_xu) {
    checkCudaErrors(cudaFree(m_d_xu));
    m_d_xu = nullptr;
  }
  if (m_d_xl) {
    checkCudaErrors(cudaFree(m_d_xl));
    m_d_xl = nullptr;
  }
  if (m_d_nbd) {
    checkCudaErrors(cudaFree(m_d_nbd));
    m_d_nbd = nullptr;
  }
  m_deviceBuffersAllocated = false;
}

template <typename T>
typename LBFGS<T>::Result LBFGS<T>::handle(ObjFuncBase<T> &func, T *x0, const Options &opts)
{
  memset(&m_lbfgsb_options, 0, sizeof(m_lbfgsb_options));
  memset(&m_summary, 0, sizeof(m_summary));

  lbfgsbcuda::lbfgsbdefaultoption<T>(m_lbfgsb_options);
  m_lbfgsb_options.mode = ((opts.runOnDevice) ? LCM_CUDA : LCM_NO_ACCELERATION);
  m_lbfgsb_options.eps_f = static_cast<T>(opts.funcTol);
  m_lbfgsb_options.eps_g = static_cast<T>(opts.gradTol);
  m_lbfgsb_options.eps_x = static_cast<T>(opts.stepTol);
  m_lbfgsb_options.max_iteration = opts.maxIter;
  m_lbfgsb_options.hessian_approximate_dimension = m_historySize;

  T minimal_f = std::numeric_limits<T>::max();
  m_state.m_funcgrad_callback = [&func,&minimal_f](T* x, T& f, T* g,
                                                   const cudaStream_t& stream,
                                                   const LBFGSB_CUDA_SUMMARY<T>& summary) {
    f = func(x, g);
    minimal_f = std::min(f,minimal_f);
    return 0;
  };

  if (opts.verbose) {
    if (opts.runOnDevice) {
      m_state.m_after_iteration_callback = [this](T* x, T& f, T* g,
                                                  const cudaStream_t& stream,
                                                  const LBFGSB_CUDA_SUMMARY<T>& summary) {
        T sum = T(0);
        std::vector<T> h_g(m_dim);
        checkCudaErrors(cudaMemcpy(h_g.data(),g,m_dim*sizeof(T),cudaMemcpyDeviceToHost));
        for (int i=0; i<m_dim; ++i) sum += h_g[i]*h_g[i];
        auto grad_norm = std::sqrt(sum);
        std::cout << "Iter: " << summary.num_iteration << " func= " << f << " ||∇func||= " << grad_norm << std::endl;
        return 0;
      };
    } else {
      m_state.m_after_iteration_callback = [this](T* x, T& f, T* g,
                                                  const cudaStream_t& stream,
                                                  const LBFGSB_CUDA_SUMMARY<T>& summary) {
        T sum = T(0);       
        for (int i=0; i<m_dim; ++i) sum += g[i]*g[i];
        auto grad_norm = std::sqrt(sum);
        std::cout << "Iter: " << summary.num_iteration << " func= " << f << " ||∇func||= " << grad_norm << std::endl;
        return 0;
      };
    }
  }

  std::vector<T> xu(m_dim,0.0), xl(m_dim,0.0);
  std::vector<int> nbd(m_dim,0);
  if (opts.lowerBound.size() > 0) {
    std::copy(opts.lowerBound.begin(), opts.lowerBound.end(), xl.begin());
    std::fill(nbd.begin(),nbd.end(),1);     // only lower bound
  }
  if (opts.upperBound.size() > 0) {
    std::copy(opts.upperBound.begin(), opts.upperBound.end(), xu.begin());
    if (opts.lowerBound.size() > 0) {
      std::fill(nbd.begin(),nbd.end(),2);   // both bounds
    } else {
      std::fill(nbd.begin(),nbd.end(),3);   // only upper bound
    }
  }

  typename LBFGS<T>::Result results;
  if (opts.runOnDevice) {
    cublasStatus_t stat = cublasCreate(&(m_state.m_cublas_handle));
    if (CUBLAS_STATUS_SUCCESS != stat) {
      throw InternalError("LBFGS<T>::handle: fail to allocate cublas handle !");
    }
    allocateDeviceBuffers();
    checkCudaErrors(cudaMemcpy(m_d_xl,xl.data(),m_dim*sizeof(T),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_d_xu,xu.data(),m_dim*sizeof(T),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_d_nbd,nbd.data(),m_dim*sizeof(int),cudaMemcpyHostToDevice));
    lbfgsbcuda::lbfgsbminimize<T>(m_dim, m_state, m_lbfgsb_options, x0, m_d_nbd, m_d_xl, m_d_xu, m_summary);
  } else {
    lbfgsbcuda::lbfgsbminimize<T>(m_dim, m_state, m_lbfgsb_options, x0, nbd.data(), xl.data(), xu.data(), m_summary);
  }
  if (opts.verbose) {
    std::cout << "lbfgsbcuda summary info: " << m_summary.info << std::endl;
  }
  // Meaning of info:
  // ================
  // -2 : Unknown internal error
  // -1 : Wrong parameters were specified (invalid input)
  //  0 : Interrupted by user (via callback returning non-zero)
  //  1 : Converged: relative function decreasing ≤ `EpsF`
  //  2 : Converged: step size ≤ `EpsX`
  //  3 : Converged: gradient norm ≤ `EpsG` (desired outcome)
  //  4 : Converged: gradient norm ≤ `EpsG` (desired outcome)
  //  5 : Max iterations reached without convergence
  results.converged = (m_summary.info > 0 and m_summary.info < 5);
  results.iterations = m_summary.num_iteration;
  results.finalGradientNorm = m_summary.residual_g;
  results.optimalValue = minimal_f;
  return results;
}

// Instantiate for float and double
template class LBFGS<float>;
template class LBFGS<double>;

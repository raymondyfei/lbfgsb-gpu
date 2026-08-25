#pragma once

#include <vector>
#include <limits>
#include "culbfgsb/culbfgsb.h"

//////////////////////////////////////////////////////////////////////////////
// ObjFuncBase - Abstract base class for objective functions
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Abstract base class for objective functions.
 * @tparam T Type of vector elements
 * @tparam R Return type (floating point, non-complex)
 */
template <typename T> 
class ObjFuncBase {
public:
  ObjFuncBase(int dim) : m_dim(dim) {}
  virtual ~ObjFuncBase() = default;

  /**
   * @brief Evaluate function value and gradient simultaneously.
   * @param x Input point (array of size dim)
   * @param grad Output gradient (array of size dim, can be nullptr if not needed)
   * 
   * When using handle method with option runOnDevice=true, both x and grad should point 
   * to device buffer and the method is expected to run on the device.
   * 
   * @return Function value at x
   */
  virtual T operator()(const T *x, T *grad) = 0;

  inline int dim() const { return m_dim; }

protected:
  int m_dim;
};

///////////////////////////////////////////////////////////////////////////////
// LBFGS - Limited-memory BFGS optimizer
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Limited-memory BFGS quasi-Newton optimizer.
 *
 * Implements the L-BFGS algorithm as described in Nocedal & Wright,
 * "Numerical Optimization", Algorithm 7.2. Uses a two-loop recursion
 * to compute the search direction without explicitly forming the
 * approximate Hessian matrix.
 *
 * @tparam T Type of vector elements (e.g., float, double)
 */
template <typename T> 
class LBFGS {
public:
  /**
   * @brief Optimization options.
   */
  struct Options {
    int maxIter = 1000;
    T gradTol = T(1e-6);
    T funcTol = T(1e-10);
    T stepTol = T(1e-10);
    bool verbose = false;
    T initialAlpha = T(1.0);
    std::vector<T> lowerBound {};  // Per-variable lower bounds (empty = no bounds)
    std::vector<T> upperBound {};  // Per-variable upper bounds (empty = no bounds)
    bool runOnDevice = false;
  };

  /**
   * @brief Optimization result.
   */
  struct Result {
    bool converged = false;
    int iterations = -1;
    T optimalValue = std::numeric_limits<T>::quiet_NaN();
    T finalGradientNorm = std::numeric_limits<T>::quiet_NaN();
  };

  /**
   * @brief Construct LBFGS optimizer.
   * @param dim Problem dimension
   * @param historySize L-BFGS history size (default: 7)
   */
  LBFGS(int dim, int historySize = 7);

  ~LBFGS();

  /**
   * @brief Run L-BFGS optimization.
   * @param func Objective function
   * @param x0 Initial guess (modified in-place to store the result)
   * @param opts Optimization options
   * 
   * When opts.runOnDevice is true, func should be a wrapper of a CUDA
   * kernel, and x0 should be a pointer to a device memory.
   * 
   * @return Result structure with optimization outcome
   */
  Result handle(ObjFuncBase<T> &func, T *x0, const Options &opts = Options());

private:
  void allocateDeviceBuffers();
  void freeDeviceBuffers();
private:
  int m_dim;
  int m_historySize;

  LBFGSB_CUDA_OPTION<T>   m_lbfgsb_options;
  LBFGSB_CUDA_STATE<T>    m_state;
  LBFGSB_CUDA_SUMMARY<T>  m_summary;

  bool                     m_deviceBuffersAllocated = false;
  T                       *m_d_xu = nullptr;
  T                       *m_d_xl = nullptr;
  int                     *m_d_nbd = nullptr;  
};


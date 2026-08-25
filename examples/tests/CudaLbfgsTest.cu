/*
 * Copyright (c) 2026
 *
 * LBFGS optimizer unit tests using Google Test.
 */

#include <eigen3/Eigen/Eigen>
#include "Lbfgs.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <random>

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

inline void __cuBlasCall(cublasStatus_t status, const char *file, const int line)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::stringstream ss;
    ss << "Cublas Error in " << file << " @ " << line << ". status code: " << status;
    std::cerr << ss.str() << "\n";
    throw InternalError(ss.str());
  }
}
#define CUBLAS_CALL(err) __cuBlasCall(err, __FILE__, __LINE__)

template <typename T>
__device__ T sqr(T x) { return x*x; }

template <typename T>
__global__ void quadratic_objective_kernel(const T *dX, const T *dX0, int dim, T *dGrad, T *dErr2)
{
  unsigned int idx = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;

  if (idx >= dim)
    return;

  T d = dX[idx] - dX0[idx];
  dErr2[idx] = sqr(d);
  dGrad[idx] = static_cast<T>(2)*d;
}

template <typename T>
class CudaQuadraticFunc : public ObjFuncBase<T> {
public:
   CudaQuadraticFunc(const std::vector<T> &x0)
     : ObjFuncBase<T>(static_cast<int>(x0.size())), x0(x0) {
     cudaMalloc((void**)&m_d_x0,this->m_dim*sizeof(T));
     cudaMemcpy(m_d_x0,x0.data(),this->m_dim*sizeof(T),cudaMemcpyHostToDevice);
     cudaMalloc((void**)&m_d_err2,this->m_dim*sizeof(T));
  }
  ~CudaQuadraticFunc() {
    cudaFree(m_d_x0);
    cudaFree(m_d_err2);
  }

  T operator()(const T *x, T *grad) override {
    const unsigned int Nx = 16;
    const unsigned int Ny = 16;
    dim3 dimBlock(Nx,Ny,1);
    unsigned int Nz = (this->m_dim - 1)/(Nx*Ny) + 1;
    dim3 dimGrid(Nz,1);
    quadratic_objective_kernel<T><<<Nz,dimBlock>>>(x,m_d_x0,this->m_dim,grad,m_d_err2);
    cudaDeviceSynchronize();
    thrust::device_ptr<T> d_thrust_ptr = thrust::device_pointer_cast(m_d_err2);
    T obj = thrust::reduce(d_thrust_ptr, d_thrust_ptr + this->m_dim);
    return obj;
  }

  const std::vector<T> &expected_minimum() const { return x0; }

private:
   std::vector<T> x0;
   T *m_d_err2;
   T *m_d_x0;
 };


template <typename T>
class CuL2ObjFunc : public ObjFuncBase<T> {
public:
  ///
  /// \fn CuL2ObjFunc constructor
  ///
  /// \param N matrix number of rows
  /// \param M matrix number of columns
  /// \param A a host pointer to the matrix elements in row-major format
  /// \param y a host pointer to the data vector
  ///
  CuL2ObjFunc(unsigned int N, unsigned int M, const T *A, const T *y);
  ~CuL2ObjFunc();

  /// The implemented pure virtual methods
  T operator()(const T* x, T* grad);
  ///
  /// \fn setAandY
  /// \brief set new matrix \f${\bf A}\f$ and RHS vector \f${\bf y}\f$
  ///
  /// Assume given pointers are on host and have the same dimensions as expected due to Ctor paramters.
  ///
  /// \param A - a host pointer to the \f${\bf A}\f$ matrix elements given in row-major order.
  /// \param y - a host pointer to the RHS vector \f${\bf y}\f$.
  ///
  void setAandY(const T* A, const T* y);

private:
  // Assume m_N >= m_M
  unsigned int        m_N;            ///< Number of A rows (not in CUBLAS)
  unsigned int        m_M;            ///< Number of A columns (not in CUBLAS)
  const T*            m_A;            ///< A matrix - row-wise concatenated
  const T*            m_y;            ///< RHS column vector (length: m_N)
  T*                  m_dA;           ///< A matrix - column-wise concatenated (on device)
  T*                  m_dY;           ///< RHS column vector (length: m_N) (on device)
  T*                  m_AxMinusY;     ///< A*x-y vector (length: m_N)
  T*                  m_sqrAxMinusY;  ///< |A*x-y|^2 
  T                   m_one;
  T                   m_minusOne;
  T                   m_zero;
  T                   m_two;
  cublasHandle_t      m_cuBlasHandle;
};


template <typename T>
__global__ void sqrKernel(const T* x, T *z, unsigned int dim)
{
	unsigned int indx = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
	if (indx >= dim) {
		return;
	}
	T xx = x[indx];
	z[indx] = xx*xx;
}

template <typename T>
void sqr(const T* x, T *z, unsigned int dim)
{
  const unsigned int Nx = 32;
  const unsigned int Ny = 32;
  dim3 dimBlock(Nx,Ny,1);
  const unsigned int Nz = (dim - 1)/(Nx*Ny) + 1;
  dim3 dimGrid(Nz,1);
	sqrKernel<<<dimGrid,dimBlock>>>(x,z,dim);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CuL2ObjFunc<T>::CuL2ObjFunc(unsigned int N, unsigned int M, const T *A, const T *y) 
	: ObjFuncBase<T>(M), m_N(N), m_M(M)
{
  static_assert(std::is_same<T,float>::value || std::is_same<T,double>::value);
	if (M > N) {
		throw InternalError("CuL2ObjFunc::CuL2ObjFunc: under determined problem !");
	}
	m_A = A;
	m_y = y;

	checkCudaErrors(cudaMalloc((void**)&m_dA,m_M*m_N*sizeof(T)));
	checkCudaErrors(cudaMalloc((void**)&m_dY,m_N*sizeof(T)));
	checkCudaErrors(cudaMalloc((void**)&m_AxMinusY,m_N*sizeof(T)));
	checkCudaErrors(cudaMalloc((void**)&m_sqrAxMinusY,m_N*sizeof(T)));

	checkCudaErrors(cudaMemcpy(m_dA,m_A,m_N*m_M*sizeof(T),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_dY,m_y,m_N*sizeof(T),cudaMemcpyHostToDevice));

	m_one = static_cast<T>(1);
	m_two = static_cast<T>(2);
	m_zero = static_cast<T>(0);
	m_minusOne = static_cast<T>(-1);
	CUBLAS_CALL(cublasCreate(&m_cuBlasHandle));
}

template <typename T>
CuL2ObjFunc<T>::~CuL2ObjFunc()
{
	checkCudaErrors(cudaFree(m_AxMinusY));
	checkCudaErrors(cudaFree(m_sqrAxMinusY));
	checkCudaErrors(cudaFree(m_dA));
	checkCudaErrors(cudaFree(m_dY));
	CUBLAS_CALL(cublasDestroy(m_cuBlasHandle));
}

//
// Due to the fact that CUBLAS assumes that matrices are column-wise
// concatenated, we deal the A matrix in a "traspose" way. i.e. we take 
// the matrix dimensions and MxN instead of NxM (any way N>M) and use
// it with a transpose operation - this provides an equivalent behavior.

template <typename T>
T CuL2ObjFunc<T>::operator()(const T* x, T* grad) 
{
	checkCudaErrors(cudaMemcpy(m_AxMinusY,m_dY,m_N*sizeof(T),cudaMemcpyDeviceToDevice));
  if constexpr (std::is_same<T,float>::value)
	  CUBLAS_CALL(cublasSgemv(m_cuBlasHandle,CUBLAS_OP_T,m_M,m_N,&m_one,m_dA,m_M,x,1,&m_minusOne,m_AxMinusY,1));
  else 
	  CUBLAS_CALL(cublasDgemv(m_cuBlasHandle,CUBLAS_OP_T,m_M,m_N,&m_one,m_dA,m_M,x,1,&m_minusOne,m_AxMinusY,1));
	sqr(m_AxMinusY,m_sqrAxMinusY,m_N);
	thrust::device_ptr<T> ptrVals(m_sqrAxMinusY);
	T sum = thrust::reduce(ptrVals,ptrVals+m_N);
  if constexpr(std::is_same<T,float>::value)
	  CUBLAS_CALL(cublasSgemv(m_cuBlasHandle,CUBLAS_OP_N,m_M,m_N,&m_two,m_dA,m_M,m_AxMinusY,1,&m_zero,grad,1));
  else 
	  CUBLAS_CALL(cublasDgemv(m_cuBlasHandle,CUBLAS_OP_N,m_M,m_N,&m_two,m_dA,m_M,m_AxMinusY,1,&m_zero,grad,1));
	return sum;
}

template <typename T>
void CuL2ObjFunc<T>::setAandY(const T* A, const T* y) 
{
	m_A = A;
	m_y = y;
	checkCudaErrors(cudaMemcpy(m_dA,m_A,m_N*m_M*sizeof(T),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_dY,m_y,m_N*sizeof(T),cudaMemcpyHostToDevice));
}


///////////////////////////////////////////////////////////////////////////////
// Test: Convergence with known solution
///////////////////////////////////////////////////////////////////////////////

TEST(CudaLbfgsTest,SmallQuadratic) {
  int dim = 2;
  std::vector<float> x0_expected = {1.0f, 2.0f};
  CudaQuadraticFunc<float> func(x0_expected);
  LBFGS<float> optimizer(2, 2);

  LBFGS<float>::Options opts;
  opts.runOnDevice = true;
  opts.maxIter = 20000;
  opts.gradTol = 1e-8f;
  opts.verbose = false;

  std::vector<float> x0(dim);
  x0[0] = 5.0f;
  x0[1] = 5.0f;

  float *d_x0 = nullptr;
  cudaMalloc((void**)&d_x0,dim*sizeof(float));
  cudaMemcpy(d_x0,x0.data(),dim*sizeof(float),cudaMemcpyHostToDevice);

  LBFGS<float>::Result result = optimizer.handle(func, d_x0, opts);

  cudaMemcpy(x0.data(),d_x0,dim*sizeof(float),cudaMemcpyDeviceToHost);


  EXPECT_NEAR(x0[0],x0_expected[0],1.0e-4);
  EXPECT_NEAR(x0[1],x0_expected[1],1.0e-4);

  EXPECT_TRUE(result.converged) << "Should converge";
  EXPECT_GT(result.iterations, 0) << "Should have at least one iteration";
  EXPECT_NEAR(result.optimalValue, 0.0f, 1e-6f) << "Optimal value should be near zero";
  EXPECT_LT(result.finalGradientNorm, opts.gradTol) << "Final gradient norm should be below tolerance";
}

TEST(CudaLbfgsTest,BigQuadratic) {
  int dim = 10000;
  std::vector<double> x0_expected(dim,1.0);
  CudaQuadraticFunc<double> func(x0_expected);
  LBFGS<double> optimizer(dim);

  LBFGS<double>::Options opts;
  opts.runOnDevice = true;
  opts.maxIter = 20000;
  opts.gradTol = 1e-9f;
  opts.verbose = false;

  std::vector<double> x0(dim,0.0);

  double *d_x0 = nullptr;
  cudaMalloc((void**)&d_x0,dim*sizeof(double));
  cudaMemcpy(d_x0,x0.data(),dim*sizeof(double),cudaMemcpyHostToDevice);

  LBFGS<double>::Result result = optimizer.handle(func, d_x0, opts);

  cudaMemcpy(x0.data(),d_x0,dim*sizeof(double),cudaMemcpyDeviceToHost);


  for (int i=0; i<dim; ++i) {
    EXPECT_NEAR(x0[0],x0_expected[i],1.0e-4);
  }

  EXPECT_TRUE(result.converged) << "Should converge";
  EXPECT_GT(result.iterations, 0) << "Should have at least one iteration";
  EXPECT_NEAR(result.optimalValue, 0.0f, 1e-6f) << "Optimal value should be near zero";
  EXPECT_LT(result.finalGradientNorm, opts.gradTol) << "Final gradient norm should be below tolerance";
}

TEST(CudaLbfgsTest, SmallOverDeterminedProblem) {
  //
  // find x minimizing ||Ax-y||_2 (over determined problem)
  //
  int dim = 2;
  std::vector<float> A {  1, -2,
                          3, -4,
                          5, 6};
  std::vector<float> y { 1, -1, 1 };

  CuL2ObjFunc<float> func(3,dim,A.data(),y.data());

  // Caclulated using np.linalg.inv(A.T@A)@(A.T@y)
  std::vector<float> expected_x0 { static_cast<float>(0.02347418), static_cast<float>(0.13615023) };

  LBFGS<float> optimizer(dim, 2);

  LBFGS<float>::Options opts;
  opts.runOnDevice = true;
  opts.maxIter = 40000;
  opts.gradTol = 1e-12;
  opts.funcTol = 1e-20;
  opts.stepTol = 1e-12;
  opts.verbose = false;

  std::vector<float> x0(dim);
  x0[0] = 5.0f;
  x0[1] = 5.0f;

  float *d_x0 = nullptr;
  cudaMalloc((void**)&d_x0,dim*sizeof(float));
  cudaMemcpy(d_x0,x0.data(),dim*sizeof(float),cudaMemcpyHostToDevice);

  LBFGS<float>::Result result = optimizer.handle(func, d_x0, opts);

  cudaMemcpy(x0.data(),d_x0,dim*sizeof(float),cudaMemcpyDeviceToHost);
  EXPECT_NEAR(x0[0], expected_x0[0], 1e-6f) << "1st optimal value should be near " << expected_x0[0];
  EXPECT_NEAR(x0[1], expected_x0[1], 1e-6f) << "1st optimal value should be near " << expected_x0[1];
}

TEST(CudaLbfgsTest, BigL2ProgramDouble) {
  //
  // find x minimizing ||Ax-y||₂²  (over determined problem)
  //
  int dim = 1000;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(dim,dim);
  Eigen::Matrix<double, Eigen::Dynamic, 1> y(dim);
  A.setRandom();
  y.setRandom();

  Eigen::Matrix<double, Eigen::Dynamic, 1> expected_x = A.lu().solve(y);

  CuL2ObjFunc<double> func(uint32_t(dim),uint32_t(dim),A.data(),y.data());

  LBFGS<double> optimizer(dim, 8);

  LBFGS<double>::Options opts;
  opts.runOnDevice = true;
  opts.maxIter = 40000;
  opts.gradTol = 1e-12;
  opts.funcTol = 1e-20;
  opts.stepTol = 1e-12;
  opts.verbose = false;

  //std::vector<double> x0(dim,0.0);
  std::vector<double> x0(expected_x.data(),expected_x.data()+dim);

  double *d_x0 = nullptr;
  cudaMalloc((void**)&d_x0,dim*sizeof(double));
  cudaMemcpy(d_x0,x0.data(),dim*sizeof(double),cudaMemcpyHostToDevice);

  LBFGS<double>::Result result = optimizer.handle(func, d_x0, opts);

  cudaMemcpy(x0.data(),d_x0,dim*sizeof(double),cudaMemcpyDeviceToHost);
  EXPECT_LT(result.finalGradientNorm,1.0e-6);
  EXPECT_LT(result.optimalValue,1.0e-6);
  for (int i=0;i <dim; ++i) {
    EXPECT_NEAR(x0[i],expected_x[i],1.0e-6);
  }

  cudaFree(d_x0);
}

TEST(CudaLbfgsTest, BigL2ProgramSingle) {
  //
  // find x minimizing ||Ax-y||_2 (over determined problem)
  //
  int dim = 100;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(dim,dim);
  Eigen::Matrix<float, Eigen::Dynamic, 1> y(dim);
  A.setRandom();
  y.setRandom();

  Eigen::Matrix<float, Eigen::Dynamic, 1> expected_x = A.lu().solve(y);

  CuL2ObjFunc<float> func(uint32_t(dim),uint32_t(dim),A.data(),y.data());

  LBFGS<float> optimizer(dim, 8);

  LBFGS<float>::Options opts;
  opts.runOnDevice = true;
  opts.maxIter = 40000;
  opts.gradTol = 1e-6f;
  opts.funcTol = 1e-20f;
  opts.stepTol = 1e-20f;
  opts.verbose = false;

  std::vector<float> x0(dim,0.0);

  float *d_x0 = nullptr;
  cudaMalloc((void**)&d_x0,dim*sizeof(float));
  cudaMemcpy(d_x0,x0.data(),dim*sizeof(float),cudaMemcpyHostToDevice);

  LBFGS<float>::Result result = optimizer.handle(func, d_x0, opts);

  cudaMemcpy(x0.data(),d_x0,dim*sizeof(float),cudaMemcpyDeviceToHost);
  EXPECT_LT(result.finalGradientNorm,1.0e-4);
  EXPECT_LT(result.optimalValue,1.0e-6);
  for (int i=0;i <dim; ++i) {
    EXPECT_NEAR(x0[i],expected_x[i],1.0e-3);
  }
  cudaFree(d_x0);
}

TEST(CudaLbfgsTest, BoxConstrainedQuadratic) {
  //
  // Minimize the separable quadratic sum((x_i - x0_expected_i)^2) subject to
  // box constraints lowerBound <= x <= upperBound, where the unconstrained
  // minimum x0_expected lies outside the box for two thirds of the
  // coordinates (one third below the lower bound, one third above the upper
  // bound), and inside the box for the remaining third. Because the
  // objective is fully separable (no cross terms between coordinates, see
  // quadratic_objective_kernel above), the constrained optimum is exactly
  // the elementwise clamp of x0_expected into [lowerBound, upperBound].
  //
  int dim = 30;

  std::vector<double> lowerBound(dim, -1.0);
  std::vector<double> upperBound(dim, 1.0);
  std::vector<double> x0_expected(dim);

  for (int i = 0; i < dim; ++i) {
    int m = i % 3;
    if (m == 0) {
      // Below the lower bound -> lower constraint should be active.
      x0_expected[i] = -5.0 - 0.1 * i;
    } else if (m == 1) {
      // Above the upper bound -> upper constraint should be active.
      x0_expected[i] = 5.0 + 0.1 * i;
    } else {
      // Inside the box -> constraint inactive, optimum == x0_expected.
      x0_expected[i] = 0.1 * ((i % 7) - 3);
    }
  }

  // Elementwise clamp: the true constrained optimum for a separable
  // quadratic under box constraints.
  std::vector<double> expected_constrained_optimum(dim);
  for (int i = 0; i < dim; ++i) {
    double v = x0_expected[i];
    double lo = lowerBound[i];
    double hi = upperBound[i];
    expected_constrained_optimum[i] = (v < lo) ? lo : ((v > hi) ? hi : v);
  }

  CudaQuadraticFunc<double> func(x0_expected);
  LBFGS<double> optimizer(dim, 8);

  LBFGS<double>::Options opts;
  opts.runOnDevice = true;
  opts.maxIter = 20000;
  opts.gradTol = 1e-9;
  opts.funcTol = 1e-12;
  opts.stepTol = 1e-12;
  opts.verbose = false;
  opts.lowerBound = lowerBound;
  opts.upperBound = upperBound;

  // Start point outside the box in both directions.
  std::vector<double> x0(dim);
  for (int i = 0; i < dim; ++i) {
    x0[i] = (i % 2 == 0) ? 4.0 : -4.0;
  }

  double *d_x0 = nullptr;
  cudaMalloc((void**)&d_x0, dim * sizeof(double));
  cudaMemcpy(d_x0, x0.data(), dim * sizeof(double), cudaMemcpyHostToDevice);

  LBFGS<double>::Result result = optimizer.handle(func, d_x0, opts);

  cudaMemcpy(x0.data(), d_x0, dim * sizeof(double), cudaMemcpyDeviceToHost);

  const double boundEps = 1e-6;
  for (int i = 0; i < dim; ++i) {
    EXPECT_GE(x0[i], lowerBound[i] - boundEps) << "x[" << i << "] should respect lower bound";
    EXPECT_LE(x0[i], upperBound[i] + boundEps) << "x[" << i << "] should respect upper bound";
    EXPECT_NEAR(x0[i], expected_constrained_optimum[i], 1e-5)
        << "x[" << i << "] should converge to the elementwise-clamped optimum";
  }

  EXPECT_TRUE(result.converged) << "Should converge";
  EXPECT_GT(result.iterations, 0) << "Should have at least one iteration";
  // culbfgsb's finalGradientNorm is the PROJECTED gradient norm (sbgnrm),
  // the algorithm's own stopping criterion, so it is expected to be small
  // even at coordinates pinned to an active bound.
  EXPECT_LT(result.finalGradientNorm, opts.gradTol)
      << "Final projected gradient norm should be below tolerance";

  double expectedOptimalValue = 0.0;
  for (int i = 0; i < dim; ++i) {
    double d = expected_constrained_optimum[i] - x0_expected[i];
    expectedOptimalValue += d * d;
  }
  EXPECT_NEAR(result.optimalValue, expectedOptimalValue, 1e-6);

  cudaFree(d_x0);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

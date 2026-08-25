/*
 * Copyright (c) 2026
 *
 * LBFGS optimizer unit tests using Google Test.
 */

#include "Lbfgs.h"
#include <cmath>
#include <complex>
#include <eigen3/Eigen/Eigen>
#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
// Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2
// Minimum at (a, a^2) with f = 0
///////////////////////////////////////////////////////////////////////////////
template <typename T>
class RosenbrockFunc : public ObjFuncBase<T> {
public:
  RosenbrockFunc(T a, T b) : ObjFuncBase<T>(2), a(a), b(b) {}

  T operator()(const T *x, T *grad) override {
    T dx = a - x[0];
    T tmp = x[1] - x[0] * x[0];
    T obj = dx * dx + b * tmp * tmp;
    grad[0] = static_cast<T>(-2) * dx - static_cast<T>(4) * b * tmp * x[0];
    grad[1] = static_cast<T>(2) * b * tmp;
    return obj;
  }
  std::pair<T, T> expected_minimum() const { return {a, a * a}; }

private:
  T a, b;
};

///////////////////////////////////////////////////////////////////////////////
// Simple quadratic: f(x) = sum((x[i] - x0[i])^2)
// Minimum at x0 with f = 0
///////////////////////////////////////////////////////////////////////////////
class QuadraticFunc : public ObjFuncBase<float> {
public:
  QuadraticFunc(const std::vector<float> &x0)
      : ObjFuncBase<float>(static_cast<int>(x0.size())), x0(x0) {}

  float operator()(const float *x, float *grad) override {
    float obj = 0.0f;
    for (int i = 0; i < m_dim; ++i) {
      float d = x[i] - x0[i];
      obj += d * d;
      grad[i] = 2.0f * d;
    }
    return obj;
  }
  const std::vector<float> &expected_minimum() const { return x0; }

private:
  std::vector<float> x0;
};

///////////////////////////////////////////////////////////////////////////////
// Beale's function: f(x,y)
// Minimum at (3, 0.5) with f = 0
///////////////////////////////////////////////////////////////////////////////
class BealeFunc : public ObjFuncBase<float> {
public:
  BealeFunc() : ObjFuncBase<float>(2) {}

  float operator()(const float *x, float *grad) override {
    float t1 = 1.5f - x[0] + x[0] * x[1];
    float t2 = 2.25f - x[0] + x[0] * x[1] * x[1];
    float t3 = 2.625f - x[0] + x[0] * x[1] * x[1] * x[1];
    float obj = t1 * t1 + t2 * t2 + t3 * t3;

    grad[0] = 2.0f * t1 * (-1.0f + x[1]) + 2.0f * t2 * (-1.0f + x[1] * x[1]) +
              2.0f * t3 * (-1.0f + x[1] * x[1] * x[1]);

    grad[1] = 2.0f * t1 * x[0] + 2.0f * t2 * (2.0f * x[0] * x[1]) +
              2.0f * t3 * (3.0f * x[0] * x[1] * x[1]);

    return obj;
  }
  std::pair<float, float> expected_minimum() const { return {3.0f, 0.5f}; }
};

///////////////////////////////////////////////////////////////////////////////
// Quadratic function with complex numbers
///////////////////////////////////////////////////////////////////////////////
// ||Ax-b||^2

template <typename T> T sqr(T x) { return x * x; }

template <typename T> class LeastSquaresFunc : public ObjFuncBase<T> {
public:
  LeastSquaresFunc(uint32_t dim,
                   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A,
                   Eigen::Matrix<T, Eigen::Dynamic, 1> &b)
      : ObjFuncBase<T>(dim), m_A(A), m_b(b) {}

  T operator()(const T *x, T *grad) override {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> xVec(x, this->m_dim);
    Eigen::Matrix<T, Eigen::Dynamic, 1> res = m_A * xVec - m_b;
    T obj = sqr(res.norm());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> gradVec(grad, this->m_dim);
    gradVec = static_cast<T>(2) * m_A.transpose() * res;
    return obj;
  }
  Eigen::Matrix<T, Eigen::Dynamic, 1> expected_minimum() const {
    return (m_A.lu().solve(m_b)).eval();
  }

private:
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_A;
  Eigen::Matrix<T, Eigen::Dynamic, 1> m_b;
};

///////////////////////////////////////////////////////////////////////////////
// Test: Rosenbrock function
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, RosenbrockTest) {
  RosenbrockFunc<float> func(1.0f, 100.0f);
  LBFGS<float> optimizer(2, 2);

  LBFGS<float>::Options opts;
  opts.maxIter = 5000;
  opts.gradTol = 1e-4f;
  opts.verbose = false;

  std::vector<float> x0 = {-1.2f, 1.0f};

  optimizer.handle(func, x0.data(), opts);

  auto [expX, expY] = func.expected_minimum();

  float error = std::sqrt((x0[0] - expX) * (x0[0] - expX) +
                          (x0[1] - expY) * (x0[1] - expY));

  EXPECT_LT(error, 0.1f) << "Rosenbrock result should be close to minimum, error=" << error;
}

///////////////////////////////////////////////////////////////////////////////
// Test: Simple quadratic function
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, QuadraticTest) {
  std::vector<float> x0_expected = {2.0f, -1.0f, 3.5f};
  QuadraticFunc func(x0_expected);
  LBFGS<float> optimizer(3, 3);

  LBFGS<float>::Options opts;
  opts.maxIter = 10000;
  opts.gradTol = 1e-6f;
  opts.verbose = false;

  std::vector<float> x0(3);
  x0[0] = 0.0f;
  x0[1] = 0.0f;
  x0[2] = 0.0f;

  LBFGS<float>::Result result =
      optimizer.handle(func, x0.data(), opts);

  float error = 0.0f;
  for (int i = 0; i < 3; ++i) {
    error += (x0[i] - x0_expected[i]) * (x0[i] - x0_expected[i]);
  }
  error = std::sqrt(error);

  EXPECT_TRUE(result.converged) << "Quadratic optimization should converge";
  EXPECT_LT(error, 1e-4f) << "Quadratic result should be close to minimum";
}

///////////////////////////////////////////////////////////////////////////////
// Test: Beale's function
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, BealeTest) {
  BealeFunc func;
  LBFGS<float> optimizer(2, 2);

  LBFGS<float>::Options opts;
  opts.maxIter = 10000;
  opts.gradTol = 1e-3f;
  opts.verbose = false;

  std::vector<float> x0 = {1.0f, 1.0f};

  optimizer.handle(func, x0.data(), opts);

  auto [expX, expY] = func.expected_minimum();

  float error = std::sqrt((x0[0] - expX) * (x0[0] - expX) +
                          (x0[1] - expY) * (x0[1] - expY));

  EXPECT_LT(error, 0.5f) << "Beale result should be close to minimum, error=" << error;
}

///////////////////////////////////////////////////////////////////////////////
// Test: Convergence with known solution
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, ConvergenceTest) {
  std::vector<float> x0_expected = {1.0f, 2.0f};
  QuadraticFunc func(x0_expected);
  LBFGS<float> optimizer(2, 2);

  LBFGS<float>::Options opts;
  opts.maxIter = 20000;
  opts.gradTol = 1e-8f;
  opts.verbose = false;

  std::vector<float> x0(2);
  x0[0] = 5.0f;
  x0[1] = 5.0f;

  LBFGS<float>::Result result = optimizer.handle(func, x0.data(), opts);

  EXPECT_TRUE(result.converged) << "Should converge";
  EXPECT_GT(result.iterations, 0) << "Should have at least one iteration";
  EXPECT_NEAR(result.optimalValue, 0.0f, 1e-6f)
      << "Optimal value should be near zero";
  EXPECT_LT(result.finalGradientNorm, opts.gradTol)
      << "Final gradient norm should be below tolerance";
}

///////////////////////////////////////////////////////////////////////////////
// Test: History size parameter
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, HistorySizeTest) {
  std::vector<float> x0_expected = {1.0f, 2.0f};
  QuadraticFunc func(x0_expected);

  // Test with small history size
  LBFGS<float> optimizer1(2, 1);
  LBFGS<float>::Options opts;
  opts.maxIter = 10000;
  opts.gradTol = 1e-6f;
  opts.verbose = false;

  std::vector<float> x0(2);
  x0[0] = 0.0f;
  x0[1] = 0.0f;

  LBFGS<float>::Result result1 =
      optimizer1.handle(func, x0.data(), opts);

  // Test with large history size
  LBFGS<float> optimizer2(2, 2);
  std::vector<float> x0_2(2);
  x0_2[0] = 0.0f;
  x0_2[1] = 0.0f;

  LBFGS<float>::Result result2 =
      optimizer2.handle(func, x0_2.data(), opts);

  EXPECT_TRUE(result1.converged) << "Small history size should converge";
  EXPECT_TRUE(result2.converged) << "Large history size should converge";
}

///////////////////////////////////////////////////////////////////////////////
// Test: Double type
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, DoubleTypeTest) {
  class DoubleRosenbrock : public ObjFuncBase<double> {
  public:
    DoubleRosenbrock() : ObjFuncBase<double>(2) {}

    double operator()(const double *x, double *grad) override {
      double dx = 1.0 - x[0];
      double tmp = x[1] - x[0] * x[0];
      double obj = dx * dx + 100.0 * tmp * tmp;
      grad[0] = -2.0 * dx - 400.0 * tmp * x[0];
      grad[1] = 200.0 * tmp;
      return obj;
    }
  };

  DoubleRosenbrock func;
  LBFGS<double> optimizer(2, 2);

  LBFGS<double>::Options opts;
  opts.maxIter = 10000;
  opts.gradTol = 1e-8;
  opts.verbose = false;

  std::vector<double> x0 = {-1.2, 1.0};

  optimizer.handle(func, x0.data(), opts);

  EXPECT_NEAR(x0[0], 1.0, 0.1) << "x0 should be close to 1.0";
  EXPECT_NEAR(x0[1], 1.0, 0.1) << "x1 should be close to 1.0";
}

///////////////////////////////////////////////////////////////////////////////
// Test: Bound-constrained optimization (bounds contain the minimum)
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, BoundConstrainedTest) {
  // Quadratic with minimum at (1, 2), within bounds
  std::vector<float> x0_expected = {1.0f, 2.0f};
  QuadraticFunc func(x0_expected);
  LBFGS<float> optimizer(2, 2);

  LBFGS<float>::Options opts;
  opts.maxIter = 10000;
  opts.gradTol = 1e-6f;
  opts.verbose = false;
  opts.lowerBound = {-5.0f, -5.0f};
  opts.upperBound = {5.0f, 5.0f};

  std::vector<float> x0(2);
  x0[0] = 0.0f;
  x0[1] = 0.0f;

  LBFGS<float>::Result result = optimizer.handle(func, x0.data(), opts);

  EXPECT_TRUE(result.converged) << "Bound-constrained optimization should converge";
  
  // Check that the solution is close to the expected minimum
  float error = std::sqrt((x0[0] - x0_expected[0]) * (x0[0] - x0_expected[0]) +
                          (x0[1] - x0_expected[1]) * (x0[1] - x0_expected[1]));
  EXPECT_LT(error, 1e-4f) << "Solution should be close to minimum";

  // Check that bounds are respected
  EXPECT_GE(x0[0], opts.lowerBound[0]) << "x0 should respect lower bound";
  EXPECT_LE(x0[0], opts.upperBound[0]) << "x0 should respect upper bound";
  EXPECT_GE(x0[1], opts.lowerBound[1]) << "x1 should respect lower bound";
  EXPECT_LE(x0[1], opts.upperBound[1]) << "x1 should respect upper bound";
}

///////////////////////////////////////////////////////////////////////////////
// Test: Bound-constrained Rosenbrock function
// The unconstrained minimum at (1, 1) is outside the bounds [-2, 0.5] x [-1, 1.5]
// The constrained minimum lies on the boundary where x = 0.5, y = 0.25.
// This tests that bound constraints are properly enforced and the optimizer
// converges to the constrained solution.
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, BoundConstrainedRosenbrockTest) {
  RosenbrockFunc<double> func(1.0, 100.0);
  LBFGS<double> optimizer(2, 2);

  LBFGS<double>::Options opts;
  opts.maxIter = 100000;
  opts.gradTol = 1e-6;
  opts.verbose = false;
  opts.lowerBound = {-2.0, -1.0};
  opts.upperBound = {0.5, 1.5};

  // Start from a feasible point
  std::vector<double> x0 = {-1.5, 1.0};
  std::vector<double> g(2);

  // Record initial objective value
  double f0 = func(x0.data(),g.data());

  auto results = optimizer.handle(func, x0.data(), opts);

  // Verify bounds are respected
  EXPECT_GE(x0[0], opts.lowerBound[0] - 1e-6) << "x should respect lower bound";
  EXPECT_LE(x0[0], opts.upperBound[0] + 1e-6) << "x should respect upper bound";
  EXPECT_GE(x0[1], opts.lowerBound[1] - 1e-6) << "y should respect lower bound";
  EXPECT_LE(x0[1], opts.upperBound[1] + 1e-6) << "y should respect upper bound";

  // The optimizer should have reduced the objective function value
  EXPECT_LT(results.optimalValue, f0) << "Optimized value should be less than initial value";

  // The constrained minimum is at (0.5, 0.25) where f = 0.25
  // The optimizer should converge close to this point
  EXPECT_NEAR(x0[0], 0.5, 1.0e-5) << "x should be near the upper bound (0.5)";
  EXPECT_NEAR(x0[1], 0.25, 1.0e-5) << "y should be near x^2 = 0.25 for the constrained minimum";
}

///////////////////////////////////////////////////////////////////////////////
// Test: Partial bounds (only lower or only upper)
///////////////////////////////////////////////////////////////////////////////
TEST(LbfgsTest, PartialBoundsTest) {
  // Quadratic with minimum at (-2, 3), only lower bound on x0
  std::vector<float> x0_expected = {-2.0f, 3.0f};
  QuadraticFunc func(x0_expected);
  LBFGS<float> optimizer(2, 2);

  LBFGS<float>::Options opts;
  opts.maxIter = 10000;
  opts.gradTol = 1e-6f;
  opts.verbose = false;
  opts.lowerBound = {-5.0f, -100.0f};  // Only lower bound matters
  opts.upperBound = {};                 // No upper bound

  std::vector<float> x0(2);
  x0[0] = 0.0f;
  x0[1] = 0.0f;

  LBFGS<float>::Result result =
      optimizer.handle(func, x0.data(), opts);

  EXPECT_TRUE(result.converged)
      << "Optimization with partial bounds should converge";

  float error = std::sqrt((x0[0] - x0_expected[0]) * (x0[0] - x0_expected[0]) +
                          (x0[1] - x0_expected[1]) * (x0[1] - x0_expected[1]));
  EXPECT_LT(error, 1e-4f) << "Solution should be close to minimum with partial bounds";
}

TEST(LbfgsTest, LeastSquaresTest) {
  uint32_t dim = 4;
  Eigen::MatrixXf A(dim, dim);
  A.setRandom();
  Eigen::VectorXf b(dim);
  b.setRandom();

  LeastSquaresFunc<float> func(dim, A, b);
  LBFGS<float> optimizer(dim, 4);

  LBFGS<float>::Options opts;
  opts.maxIter = 10000;
  opts.gradTol = 1e-9f;
  opts.funcTol = 1e-15f;
  opts.stepTol = 1e-12f;
  opts.verbose = false;

  Eigen::VectorXf x0(dim);
  x0.setRandom();

  LBFGS<float>::Result result = optimizer.handle(func, x0.data(), opts);

  EXPECT_TRUE(result.converged) << "Least squares optimization should converge";
  auto expected = func.expected_minimum();
  auto error =
      (Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>(x0.data(), dim) -
       expected)
          .norm() /
      expected.norm();
  EXPECT_LT(error, 1e-6f) << "Least squares result should be close to minimum";
}

TEST(LbfgsTest, LeastSquaresLargeTest) {
  std::srand(12345);
  using VectorMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
  uint32_t dim = 1000;
  Eigen::MatrixXd A(dim, dim);
  A.setRandom();
  Eigen::VectorXd b(dim);
  b.setRandom();

  LeastSquaresFunc<double> func(dim, A, b);
  LBFGS<double> optimizer(dim, 20);

  LBFGS<double>::Options opts;
  opts.maxIter = 40000;
  opts.gradTol = 1e-12;
  opts.funcTol = 1e-20;
  opts.stepTol = 1e-12;
  opts.verbose = false;

  Eigen::VectorXd x0(dim);
  x0.setRandom();

  LBFGS<double>::Result result = optimizer.handle(func, x0.data(), opts);

  EXPECT_TRUE(result.converged) << "Least squares optimization should converge";
  auto expected = func.expected_minimum();
  auto error = (VectorMap(x0.data(), dim) - expected).norm()/expected.norm();
  EXPECT_LT(error, 1e-6f) << "Least squares result should be close to minimum";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

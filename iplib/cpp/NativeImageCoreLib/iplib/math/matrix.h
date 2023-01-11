#pragma once

#include <iplib/common.h>
#include <math.h>

namespace ip
{
	// =============================================================================================================================================

	template <class T>
	class Matrix
	{
	public:
		Matrix();
		Matrix(int rows, int cols);
		Matrix(const Matrix &other);
		Matrix(Matrix &&other);
		~Matrix();

		Matrix<T>& operator = (const Matrix<T> &other);
		Matrix<T>& operator = (Matrix<T> &&other);

		static Matrix<T> Identity(int size);
		static Matrix<T> Zero(int rows, int cols);

		int NumRows() const;
		int NumColumns() const;
		
		T operator() (int row, int column) const;
		T& operator() (int row, int column);

		operator bool() const;

		Matrix<T> GaussSolve(const Matrix<T> &right) const;
		Matrix<T> RotateSolve(const Matrix<T> &right) const;

	private:
		T *data;
		int rows, cols;

	private:
		class LinearEquationSolver;
	};

	template <class T>
	Matrix<T> operator * (const Matrix<T> & A, const Matrix<T> & B);

	// ---------------------------------------------------------------------------------------------------------------------------------------------

	template <class T>
	class Matrix<T>::LinearEquationSolver
	{
	public:
		LinearEquationSolver(const Matrix<T> & left, const Matrix<T> & right);

		Matrix<T> GaussSolve();
		Matrix<T> RotateSolve();

	private:
		Matrix<T> A, B;

		bool ForwardGauss();
		void BackwardGauss();
		bool ForwardRotate();
	};

	// =============================================================================================================================================

	template<class T>
	Matrix<T>::Matrix()
		: rows(0), cols(0), data(nullptr) {}

	template<class T>
	Matrix<T>::Matrix(int rows, int cols)
		: rows(rows), cols(cols)
	{
		check(rows > 0 && cols > 0);
		data = new T[rows * cols];
	}

	template<class T>
	Matrix<T>::Matrix(const Matrix & other)
		: rows(other.rows), cols(other.cols)
	{
		data = new T[rows * cols];
		for (int i = 0; i < rows * cols; i++)
			data[i] = other.data[i];
	}

	template<class T>
	Matrix<T>::Matrix(Matrix && other)
		: rows(other.rows), cols(other.cols), data(other.data)
	{
		other.data = nullptr;
	}

	template<class T>
	Matrix<T>::~Matrix()
	{
		delete[] data;
	}

	template<class T>
	Matrix<T>& Matrix<T>::operator = (const Matrix<T> &other)
	{
		delete[] data;

		rows = other.rows;
		cols = other.cols;
		data = new T[rows * cols];

		for (int i = 0; i < rows * cols; i++)
			data[i] = other.data[i];
		return *this;
	}

	template<class T>
	Matrix<T>& Matrix<T>::operator = (Matrix<T> &&other)
	{
		delete[] data;

		rows = other.rows;
		cols = other.cols;
		data = other.data;

		other.data = nullptr;

		return *this;
	}

	template<class T>
	Matrix<T> Matrix<T>::Identity(int size)
	{
		Matrix<T> res = Zero(size, size);

		for (int i = 0; i < size; i++)
			res(i, i) = (T)1.0;

		return res;
	}

	template<class T>
	Matrix<T> Matrix<T>::Zero(int rows, int cols)
	{
		Matrix<T> res(rows, cols);

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				res(i, j) = (T)0.0;

		return res;
	}

	template<class T>
	inline int Matrix<T>::NumRows() const
	{
		return rows;
	}

	template<class T>
	inline int Matrix<T>::NumColumns() const
	{
		return cols;
	}

	template<class T>
	inline T Matrix<T>::operator()(int row, int column) const
	{
		dbgcheck(row >= 0 && row < rows && column >= 0 && column < cols);
		return data[row * cols + column];
	}

	template<class T>
	inline T & Matrix<T>::operator()(int row, int column)
	{
		dbgcheck(row >= 0 && row < rows && column >= 0 && column < cols);
		return data[row * cols + column];
	}

	template<class T>
	inline Matrix<T>::operator bool() const
	{
		return (data != nullptr);
	}
	
	template<class T>
	Matrix<T> Matrix<T>::GaussSolve(const Matrix<T> & right) const
	{
		return Matrix<T>::LinearEquationSolver(*this, right).GaussSolve();
	}

	template<class T>
	Matrix<T> Matrix<T>::RotateSolve(const Matrix<T> & right) const
	{
		return Matrix<T>::LinearEquationSolver(*this, right).RotateSolve();
	}

	template<class T>
	Matrix<T> operator * (const Matrix<T> & A, const Matrix<T> & B)
	{
		check(A.NumColumns() == B.NumRows());

		Matrix<T> res(A.NumRows(), B.NumColumns());

		for (int i = 0; i < A.NumRows(); i++)
			for (int j = 0; j < B.NumColumns(); j++)
			{
				T v = A(i, 0) * B(0, j);
				
				for (int k = 1; k < A.NumColumns(); k++)
					v += A(i, k) * B(k, j);

				res(i, j) = v;
			}

		return std::move(res);
	}

	// ---------------------------------------------------------------------------------------------------------------------------------------------

	template<class T>
	Matrix<T>::LinearEquationSolver::LinearEquationSolver(const Matrix<T> & left, const Matrix<T> & right)
		: A(left), B(right)
	{
		check(A.rows == B.rows);
		check(A.rows == A.cols);
	}

	template<class T>
	Matrix<T> Matrix<T>::LinearEquationSolver::GaussSolve()
	{
		if (!ForwardGauss())
			return Matrix<T>();

		BackwardGauss();

		return std::move(B);
	}

	template<class T>
	Matrix<T> Matrix<T>::LinearEquationSolver::RotateSolve()
	{
		if (!ForwardRotate())
			return Matrix<T>();

		BackwardGauss();

		return std::move(B);
	}

	template<class T>
	bool Matrix<T>::LinearEquationSolver::ForwardGauss()
	{
		for (int i = 0; i < A.rows; i++)
		{
			int max_idx = i;
			for (int j = i + 1; j < A.rows; j++)
				if (fabs(A(j, i)) > fabs(A(max_idx, i)))
					max_idx = j;

			if (fabs(A(max_idx, i)) < (T)1e-16)
				return false;

			// Swap rows
			if (max_idx > i)
			{
				for (int j = i; j < A.cols; j++)
				{
					T tmp = A(i, j);
					A(i, j) = A(max_idx, j);
					A(max_idx, j) = tmp;
				}

				for (int j = 0; j < B.cols; j++)
				{
					T tmp = B(i, j);
					B(i, j) = B(max_idx, j);
					B(max_idx, j) = tmp;
				}
			}

			// Normalize i-th row
			T q = (T)1.0 / A(i, i);

			for (int j = i; j < A.cols; j++)
				A(i, j) *= q;

			for (int j = 0; j < B.cols; j++)
				B(i, j) *= q;

			// Subtract i-th row
			for (int k = i + 1; k < A.rows; k++)
			{
				q = A(k, i);

				for (int j = i; j < A.cols; j++)
					A(k, j) -= A(i, j) * q;

				for (int j = 0; j < B.cols; j++)
					B(k, j) -= B(i, j) * q;
			}
		}

		return true;
	}

	template<class T>
	void Matrix<T>::LinearEquationSolver::BackwardGauss()
	{
		// Back step
		for (int i = A.rows - 1; i >= 0; i--)
		{
			for (int k = 0; k < i; k++)
			{
				T q = A(k, i);

				for (int j = 0; j < B.cols; j++)
					B(k, j) -= B(i, j) * q;

				A(k, i) = (T)0.0;
			}
		}
	}
	
	template<class T>
	bool Matrix<T>::LinearEquationSolver::ForwardRotate()
	{
		for (int i = 0; i < A.rows; i++)
		{
			for (int j = i + 1; j < A.rows; j++)
			{
				T a = A(i, i);
				T b = A(j, i);
				T d = sqrt(a * a + b * b);

				if (d < (T)1e-16)
					continue;

				a /= d;
				b /= d;

				// Rotate rows
				for (int k = i; k < A.cols; k++)
				{
					T v1 = A(i, k);
					T v2 = A(j, k);

					A(i, k) = v1 * a + v2 * b;
					A(j, k) = v2 * a - v1 * b;
				}

				for (int k = 0; k < B.cols; k++)
				{
					T v1 = B(i, k);
					T v2 = B(j, k);

					B(i, k) = v1 * a + v2 * b;
					B(j, k) = v2 * a - v1 * b;
				}
			}

			// Normalize row
			if (fabsf(A(i, i)) < (T)1e-16)
				return false;

			T q = (T)1.0 / A(i, i);

			for (int k = i; k < A.cols; k++)
				A(i, k) *= q;

			for (int k = 0; k < B.cols; k++)
				B(i, k) *= q;
		}

		return true;
	}
}
#pragma once

#include "matrix.h"

namespace ip
{
	// ==================================================================================================

	class QuadraticOptimization
	{
	public:

		template <typename T>
		void AddData(T *q, int len, float v);

		template <typename T, size_t L>
		void AddData(T(&q)[L], float v);

		template <typename T>
		bool CalcOptimizedCoefficients(T *target) const;

		void Reset();
		int Count() const;
		int GetVectorLength() const;

	private:
		Matrix<double> A, B;
		int N = 0;
	};

	// ==================================================================================================

	template <typename T>
	void QuadraticOptimization::AddData(T *q, int len, float v)
	{
		if (!A)
		{
			A = Matrix<double>::Zero(len, len);
			B = Matrix<double>::Zero(len, 1);
		}

		dbgcheck(A.NumColumns() == len);

		for (int j = 0; j < len; j++)
		{
			check(q[j] == q[j]);

			for (int i = 0; i < len; i++)
				A(j, i) += q[i] * q[j];

			B(j, 0) += v * q[j];
		}

		N++;
	}

	template <typename T, size_t L>
	void QuadraticOptimization::AddData(T (&q)[L], float v)
	{
		this->AddData((T*)q, (int)L, v);
	}

	template<typename T>
	bool QuadraticOptimization::CalcOptimizedCoefficients(T * target) const
	{
		Matrix<double> X = A.GaussSolve(B);

		if (X)
		{
			for (int i = 0; i < X.NumRows(); i++)
				target[i] = (T)X(i, 0);

			return true;
		}
		else
		{
			return false;
		}
	}

}
#include "../../iplib/image/variational/varmethods.h"

#include <cmath>

namespace ip
{
	const float sqrt1_2 = 1.0f / std::sqrtf(2.0f);

	/* inline static float sign(float x)
	{
		return x > 0.0f ? 1.0f : -1.0f;
	} */

	#pragma region ConvolveClass

	class ConvolveClass
	{
	public:
		ConvolveClass(const Image<float> &img, const Image<float> &kernel, Image<float> &dst);
		void Perform();

	private:
		const Image<float> &img;
		const Image<float> &kernel;
		Image<float> &dst;
		int rx, ry;

	private:
		void ProcessCheckedLine(int y);
		void ProcessUncheckedLine(int y);
		inline void ProcessUncheckedLineBorder(int x, int y);
	};

	ConvolveClass::ConvolveClass(const Image<float> &img, const Image<float> &kernel, Image<float> &dst)
		: img(img), kernel(kernel), dst(dst), rx(kernel.Width() / 2), ry(kernel.Height() / 2) {}

	void ConvolveClass::ProcessCheckedLine(int y)
	{
		for (int x = 0; x < dst.Width(); x++)
		{
			float s = 0.0f;

			for (int j = 0; j < kernel.Height(); j++)
				for (int i = 0; i < kernel.Width(); i++)
					s += img((std::min)((std::max)(x + i - rx, 0), dst.Width() - 1),
					(std::min)((std::max)(y + j - ry, 0), dst.Height() - 1)) * kernel(i, j);

			dst(x, y) = s;
		}
	}

	inline void ConvolveClass::ProcessUncheckedLineBorder(int x, int y)
	{
		float s = 0.0f;

		for (int j = 0; j < kernel.Height(); j++)
			for (int i = 0; i < kernel.Width(); i++)
				s += img((std::min)((std::max)(x + i - rx, 0), dst.Width() - 1),
					y + j - ry) * kernel(i, j);

		dst(x, y) = s;
	}

	void ConvolveClass::ProcessUncheckedLine(int y)
	{
		int x = 0;

		while (x < rx)
			ProcessUncheckedLineBorder(x++, y);

		while (x < dst.Width() - rx - 7)
		{
			__m256 s = _mm256_setzero_ps();

			for (int j = 0; j < kernel.Height(); j++)
				for (int i = 0; i < kernel.Width(); i++)
				{
					__m256 p = _mm256_load_ps(img.pixeladdr(x + i - rx, y + j - ry));
					__m256 k = _mm256_broadcast_ss(kernel.pixeladdr(i, j));

					s = _mm256_add_ps(s, _mm256_mul_ps(p, k));
				}

			_mm256_store_ps(dst.pixeladdr(x, y), s);

			x += 8;

			// dst(x++, y) = s;
		}

		while (x < dst.Width())
			ProcessUncheckedLineBorder(x++, y);
	}

	void ConvolveClass::Perform()
	{
		_mm256_zeroall();

		Parallel::For(0, dst.Height(), [this](int y)
		{
			if (y < ry || y >= dst.Height() - ry)
				ProcessCheckedLine(y);
			else
				ProcessUncheckedLine(y);
		});

		_mm256_zeroall();
	}

	#pragma endregion

	#pragma region TVL1Gradient

	class TVL1Gradient
	{
		const Image<float> &src;

	public:
		TVL1Gradient(const Image<float> &src)
			: src(src) {}

		float Unsafe(int x, int y)
		{
			float v = src(x, y);
			return sign(v - src(x - 1, y)) + sign(v - src(x + 1, y)) + sign(v - src(x, y - 1)) + sign(v - src(x, y + 1));
		}

		float Safe(int x, int y)
		{
			float v = src(x, y);

			float p1 = 0.0f;

			if (x > 0)
				p1 += sign(v - src(x - 1, y));

			if (x < src.Width() - 1)
				p1 += sign(v - src(x + 1, y));

			if (y > 0)
				p1 += sign(v - src(x, y - 1));

			if (y < src.Height() - 1)
				p1 += sign(v - src(x, y + 1));

			return p1;
		}
	};

	#pragma endregion

	#pragma region TVL2ForwardGradient

	class TVL2ForwardGradient
	{
		const Image<float> &src;

	public:
		TVL2ForwardGradient(const Image<float> &src)
			: src(src) {}

		float Value(int x, int y)
		{
			float v = src(x, y);
			float v1 = v - src(x + 1, y);
			float v2 = v - src(x, y + 1);
			return std::sqrtf(v1 * v1 + v2 * v2 + 1e-10f);
		}

		float Unsafe(int x, int y)
		{
			float n0 = (2.0f * src(x, y) - src(x + 1, y) - src(x, y + 1)) / Value(x, y);
			float n1 = (src(x, y) - src(x - 1, y)) / Value(x - 1, y);
			float n2 = (src(x, y) - src(x, y - 1)) / Value(x, y - 1);
			return n0 + n1 + n2;
		}

		float ValueSafe(int x, int y)
		{
			float v = src(x, y);
			float v1 = x < src.Width() - 1 ? v - src(x + 1, y) : 0.0f;
			float v2 = y < src.Height() - 1 ? v - src(x, y + 1) : 0.0f;
			return std::sqrtf(v1 * v1 + v2 * v2 + 1e-10f);
		}

		float Safe(int x, int y)
		{
			float res = 0.0f;

			if (x < src.Width() - 1)
				res += (src(x, y) - src(x + 1, y)) / ValueSafe(x, y);

			if (y < src.Height() - 1)
				res += (src(x, y) - src(x, y + 1)) / ValueSafe(x, y);

			if (x > 0)
				res += (src(x, y) - src(x - 1, y)) / ValueSafe(x - 1, y);

			if (y > 0)
				res += (src(x, y) - src(x, y - 1)) / ValueSafe(x, y - 1);
			
			return res;
		}
	};

	#pragma endregion

	#pragma region TVL2CenterGradient

	class TVL2CenterGradient
	{
		const Image<float> &src;

	public:
		TVL2CenterGradient(const Image<float> &src)
			: src(src) {}

		float Value(int x, int y)
		{
			float v1 = src(x + 1, y) - src(x - 1, y);
			float v2 = src(x, y + 1) - src(x, y - 1);
			return std::sqrtf(v1 * v1 + v2 * v2 + 1e-10f);
		}

		float Unsafe(int x, int y)
		{
			float v = src(x, y);
			float v0 = (v - src(x - 2, y)) / Value(x - 1, y);
			float v1 = (v - src(x + 2, y)) / Value(x + 1, y);
			float v2 = (v - src(x, y - 2)) / Value(x, y - 1);
			float v3 = (v - src(x, y + 2)) / Value(x, y + 1);
			return v0 + v1 + v2 + v3;
		}

		float ValueSafe(int x, int y)
		{
			float v1 = x > 0 && x < src.Width() - 1 && y >= 0 && y < src.Height() ? src(x + 1, y) - src(x - 1, y) : 0.0f;
			float v2 = x >= 0 && x < src.Width() && y > 0 && y < src.Height() - 1 ? src(x, y + 1) - src(x, y - 1) : 0.0f;
			return std::sqrtf(v1 * v1 + v2 * v2 + 1e-10f);
		}

		float Safe(int x, int y)
		{
			float v = src(x, y);
			float v0 = (v - src((std::max)(0, x - 2), y)) / ValueSafe(x - 1, y);
			float v1 = (v - src((std::min)(src.Width() - 1, x + 2), y)) / ValueSafe(x + 1, y);
			float v2 = (v - src(x, (std::max)(0, y - 2))) / ValueSafe(x, y - 1);
			float v3 = (v - src(x, (std::min)(src.Height() - 1, y + 2))) / ValueSafe(x, y + 1);
			return v0 + v1 + v2 + v3;
		}
	};

	#pragma endregion

	#pragma region BTVGradient

	class BTVGradient
	{
		const Image<float> &src;

	public:
		BTVGradient(const Image<float> &src)
			: src(src) {}

		float Unsafe(int x, int y)
		{
			float v = src(x, y);

			float p1 = sign(v - src(x - 1, y)) + sign(v - src(x + 1, y)) + sign(v - src(x, y - 1)) + sign(v - src(x, y + 1));
			float p2 = sign(v - src(x - 1, y - 1)) + sign(v - src(x + 1, y - 1)) + sign(v - src(x - 1, y + 1)) + sign(v - src(x + 1, y + 1));

			return p1 + p2 * sqrt1_2;
		}

		float Safe(int x, int y)
		{
			float v = src(x, y);

			float p1 = 0.0f;
			float p2 = 0.0f;

			for (int j = -1; j <= 1; j++)
				for (int i = -1; i <= 1; i++)
				{
					if (x + i < 0 || x + i >= src.Width() || y + j < 0 || y + j >= src.Height())
						continue;

					float c = src(x + i, y + j);

					if (i == 0 || j == 0)
						p1 += sign(v - c);
					else
						p2 += sign(v - c);
				}

			return p1 + p2 * sqrt1_2;

		}
	};

	#pragma endregion

	#pragma region BTVGradient2

	class BTVGradient2
	{
		const Image<float> &src;

	private:
		inline float Safe(int x, int y, int dx, int dy)
		{
			float p1 = src((std::min)((std::max)(x - 2 * dx, 0), src.Width() - 1), (std::min)((std::max)(y - 2 * dy, 0), src.Height() - 1));
			float p2 = src((std::min)((std::max)(x - dx, 0), src.Width() - 1), (std::min)((std::max)(y - dy, 0), src.Height() - 1));
			float p3 = src(x, y);
			float p4 = src((std::min)((std::max)(x + dx, 0), src.Width() - 1), (std::min)((std::max)(y + dy, 0), src.Height() - 1));
			float p5 = src((std::min)((std::max)(x + 2 * dx, 0), src.Width() - 1), (std::min)((std::max)(y + 2 * dy, 0), src.Height() - 1));

			return 2.0f * sign(2.0f * p3 - p2 - p4) + sign(p1 + p3 - 2.0f * p2) + sign(p5 + p3 - 2.0f * p4);
		}

		inline float Unsafe(int x, int y, int dx, int dy)
		{
			float p1 = src(x - 2 * dx, y - 2 * dy);
			float p2 = src(x - dx, y - dy);
			float p3 = src(x, y);
			float p4 = src(x + dx, y + dy);
			float p5 = src(x + 2 * dx, y + 2 * dy);

			return 2.0f * sign(2.0f * p3 - p2 - p4) + sign(p1 + p3 - 2.0f * p2) + sign(p5 + p3 - 2.0f * p4);
		}

	public:
		BTVGradient2(const Image<float> &src)
			: src(src) {}

		float Safe(int x, int y)
		{
			float v1 = Safe(x, y, 1, 0);
			float v2 = Safe(x, y, 0, 1);
			float v3 = Safe(x, y, 1, 1);
			float v4 = Safe(x, y, -1, 1);

			return v1 + v2 + (v3 + v4) * 0.5f;
		}

		float Unsafe(int x, int y)
		{
			float v1 = Unsafe(x, y, 1, 0);
			float v2 = Unsafe(x, y, 0, 1);
			float v3 = Unsafe(x, y, 1, 1);
			float v4 = Unsafe(x, y, -1, 1);

			return v1 + v2 + (v3 + v4) * 0.5f;
		}
	};

	#pragma endregion

	void VarMethods::Convolve(const Image<float> &x, const Image<float> &kernel, Image<float> &dst)
	{
		ConvolveClass(x, kernel, dst).Perform();
	}

	void VarMethods::DiscrepancyL1(const Image<float> &z, const Image<float> &u, const Image<float> &kernel, Image<float> &dst, Image<float> &tmp)
	{
		Convolve(z, kernel, tmp);

		Parallel::For(0, tmp.Height(), [&tmp, &u](int y)
		{
			for (int x = 0; x < tmp.Width(); x++)
				tmp(x, y) = tmp(x, y) > u(x, y) ? 1.0f : -1.0f;
		});

		Convolve(tmp, kernel, dst);
	}

	void VarMethods::DiscrepancyL2(const Image<float> &z, const Image<float> &u, const Image<float> &kernel, Image<float> &dst, Image<float> &tmp)
	{
		Convolve(z, kernel, tmp);

		Parallel::For(0, tmp.Height(), [&tmp, &u](int y)
		{
			for (int x = 0; x < tmp.Width(); x++)
				tmp(x, y) -= u(x, y);
		});

		Convolve(tmp, kernel, dst);
	}

	float VarMethods::CalcNormL1(const Image<float> &x)
	{
		float res = 0.0f;

		for (int j = 0; j < x.Height(); j++)
		{
			float tmp = 0.0f;

			for (int i = 0; i < x.Width(); i++)
				tmp += std::fabsf(x(i, j));

			res += tmp;
		}

		return res / (x.Width() * x.Height());
	}

	float VarMethods::CalcNormL2(const Image<float> &x)
	{
		float res = 0.0f;

		for (int j = 0; j < x.Height(); j++)
		{
			float tmp = 0.0f;

			for (int i = 0; i < x.Width(); i++)
				tmp += x(i, j) * x(i, j);

			res += tmp;
		}

		return res / (x.Width() * x.Height());
	}

	void VarMethods::NormalizeGradientL1(Image<float> &grad, float target_norm)
	{
		float norm = CalcNormL1(grad);
		float q = target_norm / norm;

		Parallel::For(0, grad.Height(), [&grad, q](int y)
		{
			for (int x = 0; x < grad.Width(); x++)
				grad(x, y) *= q;
		});
	}

	template <typename T, int Padding>
	static inline void AddGradient(const Image<float> &z, Image<float> &dst, float alpha)
	{
		check(z.Width() == dst.Width() && z.Height() == dst.Height());

		T x(z);

		for (int j = 0; j < Padding; j++)
			for (int i = 0; i < dst.Width(); i++)
				dst(i, j) += x.Safe(i, j) * alpha;

		for (int j = Padding; j < dst.Height() - Padding; j++)
		{
			for (int i = 0; i < Padding; i++)
				dst(i, j) += x.Safe(i, j) * alpha;

			for (int i = Padding; i < dst.Width() - Padding; i++)
				dst(i, j) += x.Unsafe(i, j) * alpha;

			for (int i = dst.Width() - Padding; i < dst.Width(); i++)
				dst(i, j) += x.Safe(i, j) * alpha;
		}

		for (int j = dst.Height() - Padding; j < dst.Height(); j++)
			for (int i = 0; i < dst.Width(); i++)
				dst(i, j) += x.Safe(i, j) * alpha;
	}

	// L1 gradient norm
	void VarMethods::AddGradientTVL1(const Image<float> &z, Image<float> &dst, float alpha)
	{
		AddGradient<TVL1Gradient, 1>(z, dst, alpha);
	}

	// L2 gradient, forward derivatives
	void VarMethods::AddGradientForwardTVL2(const Image<float> &z, Image<float> &dst, float alpha)
	{
		AddGradient<TVL2ForwardGradient, 1>(z, dst, alpha);
	}

	// L2 gradient, central derivatives
	void VarMethods::AddGradientCentralTVL2(const Image<float> &z, Image<float> &dst, float alpha)
	{
		AddGradient<TVL2CenterGradient, 2>(z, dst, alpha);
	}

	// BTV gradient
	void VarMethods::AddGradientBTV(const Image<float> &z, Image<float> &dst, float alpha)
	{
		AddGradient<BTVGradient, 1>(z, dst, alpha);
	}

	// Second order BTV gradient
	void VarMethods::AddGradientBTV2(const Image<float> &z, Image<float> &dst, float alpha)
	{
		AddGradient<BTVGradient2, 2>(z, dst, alpha);
	}

	void VarMethods::Subtract(Image<float> &dst, const Image<float> &src1, const Image<float> &src2, float q)
	{
		Parallel::For(0, dst.Height(), [&src1, &src2, &dst, q](int y)
		{
			for (int x = 0; x < dst.Width(); x++)
				dst(x, y) = src1(x, y) - src2(x, y) * q;
		});
	}

	void VarMethods::ApplyMomentum(Image<float> &dst, const Image<float> &cur, Image<float> &prev, float mu)
	{
		Parallel::For(0, dst.Height(), [&cur, &prev, &dst, mu](int y)
		{
			for (int x = 0; x < dst.Width(); x++)
			{
				float v = prev(x, y) * mu + cur(x, y);
				prev(x, y) = v;
				dst(x, y) -= v;
			}
		});
	}
}

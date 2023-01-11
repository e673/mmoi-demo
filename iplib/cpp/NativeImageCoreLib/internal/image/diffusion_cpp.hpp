#include "../../iplib/image/diffusion/diffusion.h"
#include <algorithm>
#include <iplib/parallel.h>
#include <immintrin.h>

namespace ip
{
	const float GQ1 = 0.20710678118654752440084436210485f;
	const float GQ2 = 0.14644660940672623779957781894758f;
	const float LQ1 = 0.70710678118654752440084436210485f;
	const float LQ2 = 6.8284271247461900976033774484194f;

	void Diffusion::PeronaMalikDivisionIteration(const ip::ImageFloat& src, ip::ImageFloat& dst, float dt, float k)
	{
		Diffusion d(src, dst, dt, k);

		for (int i = 0; i < src.Width(); i++)
			d.IteratePeronaMalikDivisionSafe(i, 0);

		ip::Parallel::For(1, src.Height() - 1, [&d](int j)
		{
			d.IteratePeronaMalikDivisionSafe(0, j);

			int i = 1;

			while (i < d.src.Width() - 8)
			{
				// CalcGradient, CalcLaplas
				__m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
				__m256 c = _mm256_loadu_ps(d.src.pixeladdr(i, j));
				__m256 one = _mm256_set1_ps(1.0f);

				__m256 v = _mm256_loadu_ps(d.src.pixeladdr(i - 1, j - 1));
				__m256 laplas = v;
				__m256 a1 = _mm256_and_ps(_mm256_sub_ps(c, v), absmask);

				v = _mm256_loadu_ps(d.src.pixeladdr(i + 1, j - 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i - 1, j + 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i + 1, j + 1));
				laplas = _mm256_mul_ps(_mm256_add_ps(laplas, v), _mm256_set1_ps(LQ1));
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				__m256 grad = _mm256_mul_ps(a1, _mm256_set1_ps(GQ2));

				v = _mm256_loadu_ps(d.src.pixeladdr(i - 1, j));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_and_ps(_mm256_sub_ps(c, v), absmask);

				v = _mm256_loadu_ps(d.src.pixeladdr(i + 1, j));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i, j - 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i, j + 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				grad = _mm256_add_ps(_mm256_mul_ps(a1, _mm256_set1_ps(GQ1)), grad);
				laplas = _mm256_sub_ps(laplas, _mm256_mul_ps(c, _mm256_set1_ps(LQ2)));
			
				// DivisonCoefficient
				grad = _mm256_div_ps(grad, _mm256_set1_ps(d.k));
				grad = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(grad, grad)));

				// Apply
				c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(d.dt), grad), laplas));
				_mm256_storeu_ps(d.dst.pixeladdr(i, j), c);

				i += 8;
			}

			while (i < d.src.Width())
			{
				d.IteratePeronaMalikDivisionSafe(i++, j);
			}
		});

		for (int i = 0; i < src.Width(); i++)
			d.IteratePeronaMalikDivisionSafe(i, src.Height() - 1);
	}

	void Diffusion::PeronaMalikExponentIteration(const ip::ImageFloat& src, ip::ImageFloat& dst, float dt, float k)
	{
		Diffusion d(src, dst, dt, k);

		for (int i = 0; i < src.Width(); i++)
			d.IteratePeronaMalikExponentSafe(i, 0);

		ip::Parallel::For(1, src.Height() - 1, [&d](int j)
		{
			d.IteratePeronaMalikExponentSafe(0, j);

			int i = 1;

			while (i < d.src.Width() - 8)
			{
				// CalcGradient, CalcLaplas
				__m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
				__m256 c = _mm256_loadu_ps(d.src.pixeladdr(i, j));
				// __m256 one = _mm256_set1_ps(1.0f);

				__m256 v = _mm256_loadu_ps(d.src.pixeladdr(i - 1, j - 1));
				__m256 laplas = v;
				__m256 a1 = _mm256_and_ps(_mm256_sub_ps(c, v), absmask);

				v = _mm256_loadu_ps(d.src.pixeladdr(i + 1, j - 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i - 1, j + 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i + 1, j + 1));
				laplas = _mm256_mul_ps(_mm256_add_ps(laplas, v), _mm256_set1_ps(LQ1));
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				__m256 grad = _mm256_mul_ps(a1, _mm256_set1_ps(GQ2));

				v = _mm256_loadu_ps(d.src.pixeladdr(i - 1, j));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_and_ps(_mm256_sub_ps(c, v), absmask);

				v = _mm256_loadu_ps(d.src.pixeladdr(i + 1, j));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i, j - 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				v = _mm256_loadu_ps(d.src.pixeladdr(i, j + 1));
				laplas = _mm256_add_ps(laplas, v);
				a1 = _mm256_add_ps(_mm256_and_ps(_mm256_sub_ps(c, v), absmask), a1);

				grad = _mm256_add_ps(_mm256_mul_ps(a1, _mm256_set1_ps(GQ1)), grad);
				laplas = _mm256_sub_ps(laplas, _mm256_mul_ps(c, _mm256_set1_ps(LQ2)));

				// DivisonCoefficient
				grad = _mm256_div_ps(grad, _mm256_set1_ps(d.k));
				grad = _mm256_exp_ps(_mm256_sub_ps(_mm256_set1_ps(0.0f), _mm256_mul_ps(grad, grad)));

				// Apply
				c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(d.dt), grad), laplas));
				_mm256_storeu_ps(d.dst.pixeladdr(i, j), c);

				i += 8;
			}

			while (i < d.src.Width())
			{
				d.IteratePeronaMalikExponentSafe(i++, j);
			}
		});

		for (int i = 0; i < src.Width(); i++)
			d.IteratePeronaMalikExponentSafe(i, src.Height() - 1);
	}

	Diffusion::Diffusion(const ip::ImageFloat& src, ip::ImageFloat& dst, float dt, float k)
		: src(src), dst(dst), dt(dt), k(k) {}

	static float DivisonCoefficient(float grad, float k)
	{
		float q = grad / k;
		return 1.0f / (1.0f + q * q);
	}

	static float ExponentCoefficient(float grad, float k)
	{
		float q = grad / k;
		return expf(-q * q);
	}

	void Diffusion::IteratePeronaMalikDivisionSafe(int x, int y)
	{
		float grad = CalcGradientSafe(x, y);
		float q = DivisonCoefficient(grad, k);
		float laplas = CalcLaplasSafe(x, y);
		dst(x, y) = src(x, y) + dt * q * laplas;
	}

	void Diffusion::IteratePeronaMalikExponentSafe(int x, int y)
	{
		float grad = CalcGradientSafe(x, y);
		float q = ExponentCoefficient(grad, k);
		float laplas = CalcLaplasSafe(x, y);
		dst(x, y) = src(x, y) + dt * q * laplas;
	}


	float Diffusion::GetSafe(int x, int y)
	{
		auto x0 = (std::max)(0, (std::min)(src.Width() - 1, x));
		auto y0 = (std::max)(0, (std::min)(src.Height() - 1, y));
		return src(x0, y0);
	}

	float Diffusion::CalcGradientSafe(int x, int y)
	{
		float p = src(x, y);
		float p1 = fabsf(p - GetSafe(x - 1, y)) + fabsf(p - GetSafe(x + 1, y)) + fabsf(p - GetSafe(x, y - 1)) + fabsf(p - GetSafe(x, y + 1));
		float p2 = fabsf(p - GetSafe(x - 1, y - 1)) + fabsf(p - GetSafe(x + 1, y - 1)) + fabsf(p - GetSafe(x - 1, y + 1)) + fabsf(p - GetSafe(x + 1, y + 1));
		return p1 * GQ1 + p2 * GQ2;
	}

	float Diffusion::CalcLaplasSafe(int x, int y)
	{
		float p = (GetSafe(x - 1, y - 1) + GetSafe(x + 1, y - 1) + GetSafe(x - 1, y + 1) + GetSafe(x + 1, y + 1)) * LQ1 +
			GetSafe(x, y - 1) + GetSafe(x, y + 1) + GetSafe(x - 1, y) + GetSafe(x + 1, y);

		return p - src(x, y) * LQ2;
	}
}
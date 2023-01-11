#include "edrfast.h"

#include <iplib/common.h>
#include <iplib/parallel.h>
#include <math.h>
#include <immintrin.h>

namespace ip
{
	const float kernel0[6] = { -0.00063f, -0.00180f, 0.02694f, 0.01716f, -0.15097f, 1.45306f };
	const float kernel1[6] = { 0.03297f, -0.04665f, -0.04484f, -0.05382f, 0.12650f, 0.58139f };
	const float kernel2[6] = { -0.03016f, -0.04952f, -0.05838f, 0.04283f, 0.45204f, 0.25095f };
	const float WeightThreshold = 1.25f;

	class EDRFast
	{
		int Width, Height;
		
		Image<float> tmp;
		Image<float> pw1, qw1, pw2, qw2;
		Image<float> w1, w2;

		// Image<float> 

		Image<float> r00, r10, r01, r11;
		ImageFloatColor c00, c10, c01, c11;

	public:
		EDRFast(int srcWidth, int srcHeight)
			: Width(srcWidth), Height(srcHeight)
			, tmp(srcWidth, srcHeight)
			, pw1(srcWidth, srcHeight)
			, qw1(srcWidth, srcHeight)
			, pw2(srcWidth, srcHeight)
			, qw2(srcWidth, srcHeight)
			, w1(srcWidth, srcHeight)
			, w2(srcWidth, srcHeight)
			, r00(srcWidth, srcHeight)
			, r10(srcWidth, srcHeight)
			, r01(srcWidth, srcHeight)
			, r11(srcWidth, srcHeight)
			, c00(srcWidth, srcHeight)
			, c01(srcWidth, srcHeight)
			, c10(srcWidth, srcHeight)
			, c11(srcWidth, srcHeight)
		{
			// No nothing
		}

	private:
		void DerivativeDiag1(const Image<float> &src, Image<float> &dst)
		{
			Parallel::For(0, Height - 1, [&src, &dst, this](int j)
			{
				static const __m256 SIGNMASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

				int Width8 = (Width - 1) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 s11 = _mm256_loadu_ps(src.pixeladdr(i + 1, j + 1));
					__m256 s00 = _mm256_load_ps(src.pixeladdr(i, j));
					__m256 diff = _mm256_sub_ps(s11, s00);
					__m256 res = _mm256_and_ps(diff, SIGNMASK);
					_mm256_store_ps(dst.pixeladdr(i, j), res);
				}

				for (int i = Width8; i < src.Width() - 1; i++)
				{
					dst(i, j) = fabsf(src(i + 1, j + 1) - src(i, j));
				}
			});
		}

		void DerivativeDiag2(const Image<float> &src, Image<float> &dst)
		{
			Parallel::For(0, Height - 1, [&src, &dst, this](int j)
			{
				static const __m256 SIGNMASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

				int Width8 = (Width - 1) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 s11 = _mm256_loadu_ps(src.pixeladdr(i + 1, j));
					__m256 s00 = _mm256_load_ps(src.pixeladdr(i, j + 1));
					__m256 diff = _mm256_sub_ps(s11, s00);
					__m256 res = _mm256_and_ps(diff, SIGNMASK);
					_mm256_store_ps(dst.pixeladdr(i, j), res);
				}

				for (int i = Width8; i < src.Width() - 1; i++)
				{
					dst(i, j) = fabsf(src(i, j + 1) - src(i + 1, j));
				}
			});
		}

		void Average3x3(const Image<float> &src, Image<float> &dst)
		{
			Parallel::For(0, Height - 3, [&src, &dst, this](int j)
			{
				int Width8 = (Width - 3) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 v00 = _mm256_load_ps(src.pixeladdr(i, j));
					__m256 v10 = _mm256_load_ps(src.pixeladdr(i + 1, j));
					__m256 v20 = _mm256_load_ps(src.pixeladdr(i + 2, j));
					__m256 v01 = _mm256_load_ps(src.pixeladdr(i, j + 1));
					__m256 v11 = _mm256_load_ps(src.pixeladdr(i + 1, j + 1));
					__m256 v21 = _mm256_load_ps(src.pixeladdr(i + 2, j + 1));
					__m256 v02 = _mm256_load_ps(src.pixeladdr(i, j + 2));
					__m256 v12 = _mm256_load_ps(src.pixeladdr(i + 1, j + 2));
					__m256 v22 = _mm256_load_ps(src.pixeladdr(i + 2, j + 2));

					__m256 s0 = _mm256_add_ps(_mm256_add_ps(v00, v10), v20);
					__m256 s1 = _mm256_add_ps(_mm256_add_ps(v01, v11), v21);
					__m256 s2 = _mm256_add_ps(_mm256_add_ps(v02, v12), v22);

					__m256 res = _mm256_add_ps(_mm256_add_ps(s0, s1), s2);

					_mm256_store_ps(dst.pixeladdr(i, j), res);
				}

				for (int i = Width8; i < Width - 3; i++)
				{
					dst(i, j) = src(i, j) + src(i + 1, j) + src(i + 2, j) +
						src(i, j + 1) + src(i + 1, j + 1) + src(i + 2, j + 1) +
						src(i, j + 2) + src(i + 1, j + 2) + src(i + 2, j + 2);
				}
			});
		}

		static float pow6(float x)
		{
			x *= x;
			return x * x * x;
		}

		inline static float CalcWeightsFast(float p, float q)
		{
			/* if (q > p * WeightThreshold)
				return 0.0f;
			else if (p > q * WeightThreshold)
				return 1.0f;
			else
				return 0.5f; */

			float p6 = pow6(p), q6 = pow6(q);
			return (1.0f + p6) / (2.0f + p6 + q6);
		}

		inline static __m256 CalcWeightsFast(__m256 p, __m256 q)
		{
			/* static const __m256 HALF = _mm256_set1_ps(0.5f);
			static const __m256 ONE = _mm256_set1_ps(1.0f);
			static const __m256 THR = _mm256_set1_ps(WeightThreshold);

			__m256 cond_one = _mm256_cmp_ps(p, _mm256_mul_ps(q, THR), _CMP_GT_OQ);
			__m256 cond_half = _mm256_cmp_ps(q, _mm256_mul_ps(p, THR), _CMP_LT_OQ);

			return _mm256_or_ps(_mm256_and_ps(cond_one, ONE), _mm256_and_ps(cond_half, HALF)); */

			const static __m256 ONES = _mm256_set1_ps(1.0f);

			__m256 p2 = _mm256_mul_ps(p, p);
			__m256 q2 = _mm256_mul_ps(q, q);
			__m256 p6 = _mm256_add_ps(ONES, _mm256_mul_ps(_mm256_mul_ps(p2, p2), p2));
			__m256 q6 = _mm256_add_ps(ONES, _mm256_mul_ps(_mm256_mul_ps(q2, q2), q2));
			return _mm256_div_ps(p6, _mm256_add_ps(p6, q6));
		}

		static void ToWeights(const Image<float> &p, const Image<float> &q, Image<float> &w)
		{
			Parallel::For(0, p.Height(), [&p, &q, &w](int j)
			{
				int Width8 = p.Width() / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					_mm256_store_ps(w.pixeladdr(i, j), CalcWeightsFast(_mm256_load_ps(p.pixeladdr(i, j)), _mm256_load_ps(q.pixeladdr(i, j))));
				}

				for (int i = Width8; i < p.Width(); i++)
				{
					w(i, j) = CalcWeightsFast(p(i, j), q(i, j));
					/* float p6 = pow6(p(i, j)), q6 = pow6(q(i, j));
					w(i, j) = (1.0f + p6) / (2.0f + p6 + q6); */
				}
			});
		}

		void CalcWeights1(const Image<float> &src)
		{
			DerivativeDiag1(src, tmp);
			Average3x3(tmp, pw1);
			DerivativeDiag2(src, tmp);
			Average3x3(tmp, qw1);
			ToWeights(pw1, qw1, w1);
		}

		void Step00(const Image<float> &src)
		{
			Parallel::For(0, Height - 4, [&src, this](int j)
			{
				int Width8 = (Width - 4) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 v0 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i, j)), _mm256_loadu_ps(src.pixeladdr(i + 4, j))),
						                      _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i, j + 4)), _mm256_loadu_ps(src.pixeladdr(i + 4, j + 4))));

					__m256 res = _mm256_mul_ps(v0, _mm256_broadcast_ss(&kernel0[0]));

					__m256 v1a = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 1, j)), _mm256_loadu_ps(src.pixeladdr(i + 3, j))),
						                       _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i, j + 1)), _mm256_loadu_ps(src.pixeladdr(i + 4, j + 1))));

					__m256 v1b = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i, j + 3)), _mm256_loadu_ps(src.pixeladdr(i + 4, j + 3))),
						                       _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 1, j + 4)), _mm256_loadu_ps(src.pixeladdr(i + 3, j + 4))));

					res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_add_ps(v1a, v1b), _mm256_broadcast_ss(&kernel0[1])));

					__m256 v2 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 2, j)), _mm256_loadu_ps(src.pixeladdr(i, j + 2))),
						                      _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 4, j + 2)), _mm256_loadu_ps(src.pixeladdr(i + 2, j + 4))));

					res = _mm256_add_ps(res, _mm256_mul_ps(v2, _mm256_broadcast_ss(&kernel0[2])));

					__m256 v3 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(src.pixeladdr(i + 3, j + 1))),
						                      _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 1, j + 3)), _mm256_loadu_ps(src.pixeladdr(i + 3, j + 3))));

					res = _mm256_add_ps(res, _mm256_mul_ps(v3, _mm256_broadcast_ss(&kernel0[3])));

					__m256 v4 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 2, j + 1)), _mm256_loadu_ps(src.pixeladdr(i + 1, j + 2))),
					                          _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 3, j + 2)), _mm256_loadu_ps(src.pixeladdr(i + 2, j + 3))));

					res = _mm256_add_ps(res, _mm256_mul_ps(v4, _mm256_broadcast_ss(&kernel0[4])));

					__m256 v5 = _mm256_loadu_ps(src.pixeladdr(i + 2, j + 2));

					res = _mm256_add_ps(res, _mm256_mul_ps(v5, _mm256_broadcast_ss(&kernel0[5])));

					_mm256_storeu_ps(r00.pixeladdr(i + 2, j + 2), res);
				}

				for (int i = Width8; i < src.Width() - 4; i++)
				{
					float v0 = src(i, j) + src(i + 4, j) + src(i, j + 4) + src(i + 4, j + 4);
					float v1 = src(i + 1, j) + src(i + 3, j) + src(i, j + 1) + src(i + 4, j + 1) +
						src(i, j + 3) + src(i + 4, j + 3) + src(i + 1, j + 4) + src(i + 3, j + 4);
					float v2 = src(i + 2, j) + src(i, j + 2) + src(i + 4, j + 2) + src(i + 2, j + 4);
					float v3 = src(i + 1, j + 1) + src(i + 3, j + 1) + src(i + 1, j + 3) + src(i + 3, j + 3);
					float v4 = src(i + 2, j + 1) + src(i + 1, j + 2) + src(i + 3, j + 2) + src(i + 2, j + 3);
					float v5 = src(i + 2, j + 2);

					r00(i + 2, j + 2) = v0 * kernel0[0] + v1 * kernel0[1] + v2 * kernel0[2] + v3 * kernel0[3] + v4 * kernel0[4] + v5 * kernel0[5];
				}
			});
		}

		static void ToGrayScale(const ImageFloatColor &src, Image<float> &dst)
		{
			Parallel::For(0, src.Height(), [&src, &dst](int j)
			{
				int Width8 = src.Width() / 8 * 8;

				static const __m256 BLUE = _mm256_set1_ps(0.114f);
				static const __m256 GREEN = _mm256_set1_ps(0.587f);
				static const __m256 RED = _mm256_set1_ps(0.299f);

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 bgra0bgra1 = _mm256_load_ps((const float*)src.pixeladdr(i, j));
					__m256 bgra2bgra3 = _mm256_load_ps((const float*)src.pixeladdr(i + 2, j));
					__m256 bgra4bgra5 = _mm256_load_ps((const float*)src.pixeladdr(i + 4, j));
					__m256 bgra6bgra7 = _mm256_load_ps((const float*)src.pixeladdr(i + 6, j));

					__m256 b02g02b13g13 = _mm256_unpacklo_ps(bgra0bgra1, bgra2bgra3);
					__m256 r02a02r13a13 = _mm256_unpackhi_ps(bgra0bgra1, bgra2bgra3);
					__m256 b46g46b57g57 = _mm256_unpacklo_ps(bgra4bgra5, bgra6bgra7);
					__m256 r46a46r57a57 = _mm256_unpackhi_ps(bgra4bgra5, bgra6bgra7);

					__m256 b0246b1357 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(b02g02b13g13), _mm256_castps_pd(b46g46b57g57)));
					__m256 g0246g1357 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(b02g02b13g13), _mm256_castps_pd(b46g46b57g57)));
					__m256 r0246r1357 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(r02a02r13a13), _mm256_castps_pd(r46a46r57a57)));

					__m256 s0246s1357 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(b0246b1357, BLUE), _mm256_mul_ps(g0246g1357, GREEN)), _mm256_mul_ps(r0246r1357, RED));

					__m256 s0123xxxx = _mm256_unpacklo_ps(s0246s1357, _mm256_castps128_ps256(_mm256_extractf128_ps(s0246s1357, 1)));
					__m256 s4567xxxx = _mm256_unpackhi_ps(s0246s1357, _mm256_castps128_ps256(_mm256_extractf128_ps(s0246s1357, 1)));

					_mm256_store_ps(dst.pixeladdr(i, j), _mm256_insertf128_ps(s0123xxxx, _mm256_castps256_ps128(s4567xxxx), 1));
				}

				for (int i = Width8; i < src.Width(); i++)
				{
					dst(i, j) = src(i, j).ToGray();
				}
			});

		}

		void Step00(const ImageFloatColor &src)
		{
			Parallel::For(0, Height - 4, [&src, this](int j)
			{
				int Width2 = (Width - 4) / 2 * 2;

				for (int i = 0; i < Width2; i += 2)
				{
					__m256 v0 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i, j)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 4, j))),
						_mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i, j + 4)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 4, j + 4))));

					__m256 res = _mm256_mul_ps(v0, _mm256_broadcast_ss(&kernel0[0]));

					__m256 v1a = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j))), _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i, j + 1)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 4, j + 1))));

					__m256 v1b = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i, j + 3)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 4, j + 3))), _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j + 4)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j + 4))));

					res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_add_ps(v1a, v1b), _mm256_broadcast_ss(&kernel0[1])));

					__m256 v2 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j)), _mm256_loadu_ps((const float*)src.pixeladdr(i, j + 2))), _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 4, j + 2)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j + 4))));

					res = _mm256_add_ps(res, _mm256_mul_ps(v2, _mm256_broadcast_ss(&kernel0[2])));

					__m256 v3 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j + 1))), _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j + 3)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j + 3))));

					res = _mm256_add_ps(res, _mm256_mul_ps(v3, _mm256_broadcast_ss(&kernel0[3])));

					__m256 v4 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j + 1)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j + 2))), _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j + 2)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j + 3))));

					res = _mm256_add_ps(res, _mm256_mul_ps(v4, _mm256_broadcast_ss(&kernel0[4])));

					__m256 v5 = _mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j + 2));

					res = _mm256_add_ps(res, _mm256_mul_ps(v5, _mm256_broadcast_ss(&kernel0[5])));

					_mm256_storeu_ps((float*)c00.pixeladdr(i + 2, j + 2), res);
				}

				for (int i = Width2; i < src.Width() - 4; i++)
				{
					PixelFloatRGBA v0 = src(i, j) + src(i + 4, j) + src(i, j + 4) + src(i + 4, j + 4);
					PixelFloatRGBA v1 = src(i + 1, j) + src(i + 3, j) + src(i, j + 1) + src(i + 4, j + 1) +
						src(i, j + 3) + src(i + 4, j + 3) + src(i + 1, j + 4) + src(i + 3, j + 4);
					PixelFloatRGBA v2 = src(i + 2, j) + src(i, j + 2) + src(i + 4, j + 2) + src(i + 2, j + 4);
					PixelFloatRGBA v3 = src(i + 1, j + 1) + src(i + 3, j + 1) + src(i + 1, j + 3) + src(i + 3, j + 3);
					PixelFloatRGBA v4 = src(i + 2, j + 1) + src(i + 1, j + 2) + src(i + 3, j + 2) + src(i + 2, j + 3);
					PixelFloatRGBA v5 = src(i + 2, j + 2);

					c00(i + 2, j + 2) = v0 * kernel0[0] + v1 * kernel0[1] + v2 * kernel0[2] + v3 * kernel0[3] + v4 * kernel0[4] + v5 * kernel0[5];
				}
			});

			ToGrayScale(c00, r00);
		}

		void Step11(const Image<float> &src)
		{
			Parallel::For(0, src.Height() - 3, [&src, this](int j)
			{
				static const __m256 ONES = _mm256_set1_ps(1.0f);

				int Width8 = (src.Width() - 3) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 w = _mm256_load_ps(w1.pixeladdr(i, j));
					__m256 dw = _mm256_sub_ps(ONES, w);

					// 0, 3

					__m256 v0 = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i, j)), _mm256_loadu_ps(src.pixeladdr(i + 3, j + 3)));
					__m256 v3 = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 3, j)), _mm256_loadu_ps(src.pixeladdr(i, j + 3)));

					__m256 s0 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v0, w), _mm256_mul_ps(v3, dw)), _mm256_broadcast_ss(&kernel1[0]));
					__m256 s3 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v3, w), _mm256_mul_ps(v0, dw)), _mm256_broadcast_ss(&kernel1[3]));
					__m256 res = _mm256_add_ps(s0, s3);

					// 1, 2

					__m256 v1a = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 1, j)), _mm256_loadu_ps(src.pixeladdr(i, j + 1)));
					__m256 v1b = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 3, j + 2)), _mm256_loadu_ps(src.pixeladdr(i + 2, j + 3)));
					__m256 v1 = _mm256_add_ps(v1a, v1b);

					__m256 v2a = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 2, j)), _mm256_loadu_ps(src.pixeladdr(i, j + 2)));
					__m256 v2b = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 3, j + 1)), _mm256_loadu_ps(src.pixeladdr(i + 1, j + 3)));
					__m256 v2 = _mm256_add_ps(v2a, v2b);

					__m256 s1 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v1, w), _mm256_mul_ps(v2, dw)), _mm256_broadcast_ss(&kernel1[1]));
					__m256 s2 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v2, w), _mm256_mul_ps(v1, dw)), _mm256_broadcast_ss(&kernel1[2]));
					res = _mm256_add_ps(res, _mm256_add_ps(s1, s2));

					// 4, 5

					__m256 v4 = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(src.pixeladdr(i + 2, j + 2)));
					__m256 v5 = _mm256_add_ps(_mm256_loadu_ps(src.pixeladdr(i + 2, j + 1)), _mm256_loadu_ps(src.pixeladdr(i + 1, j + 2)));

					__m256 s4 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v4, w), _mm256_mul_ps(v5, dw)), _mm256_broadcast_ss(&kernel1[4]));
					__m256 s5 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v5, w), _mm256_mul_ps(v4, dw)), _mm256_broadcast_ss(&kernel1[5]));
					res = _mm256_add_ps(res, _mm256_add_ps(s4, s5));

					_mm256_storeu_ps(r11.pixeladdr(i + 1, j + 1), res);
				}

				for (int i = Width8; i < src.Width() - 3; i++)
				{
					float w = w1(i, j);
					float dw = 1.0f - w;

					float v0 = src(i, j) + src(i + 3, j + 3);
					float v1 = src(i + 1, j) + src(i, j + 1) + src(i + 3, j + 2) + src(i + 2, j + 3);
					float v2 = src(i + 2, j) + src(i, j + 2) + src(i + 3, j + 1) + src(i + 1, j + 3);
					float v3 = src(i + 3, j) + src(i, j + 3);
					float v4 = src(i + 1, j + 1) + src(i + 2, j + 2);
					float v5 = src(i + 2, j + 1) + src(i + 1, j + 2);

					float res = (v0 * w + v3 * dw) * kernel1[0] +
						        (v1 * w + v2 * dw) * kernel1[1] +
						        (v2 * w + v1 * dw) * kernel1[2] +
						        (v3 * w + v0 * dw) * kernel1[3] +
						        (v4 * w + v5 * dw) * kernel1[4] +
						        (v5 * w + v4 * dw) * kernel1[5];

					r11(i + 1, j + 1) = res;
				}
			});
		}

		void Step11(const ImageFloatColor &src)
		{
			Parallel::For(0, src.Height() - 3, [&src, this](int j)
			{
				static const __m256 ONES = _mm256_set1_ps(1.0f);

				int Width2 = (src.Width() - 3) / 2 * 2;

				for (int i = 0; i < Width2; i += 2)
				{
					__m256 w = _mm256_insertf128_ps(_mm256_broadcast_ss(w1.pixeladdr(i, j)), _mm_broadcast_ss(w1.pixeladdr(i + 1, j)), 1);
					__m256 dw = _mm256_sub_ps(ONES, w);

					// 0, 3

					__m256 v0 = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i, j)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j + 3)));
					__m256 v3 = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j)), _mm256_loadu_ps((const float*)src.pixeladdr(i, j + 3)));

					__m256 s0 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v0, w), _mm256_mul_ps(v3, dw)), _mm256_broadcast_ss(&kernel1[0]));
					__m256 s3 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v3, w), _mm256_mul_ps(v0, dw)), _mm256_broadcast_ss(&kernel1[3]));
					__m256 res = _mm256_add_ps(s0, s3);

					// 1, 2

					__m256 v1a = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j)), _mm256_loadu_ps((const float*)src.pixeladdr(i, j + 1)));
					__m256 v1b = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j + 2)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j + 3)));
					__m256 v1 = _mm256_add_ps(v1a, v1b);

					__m256 v2a = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j)), _mm256_loadu_ps((const float*)src.pixeladdr(i, j + 2)));
					__m256 v2b = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 3, j + 1)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j + 3)));
					__m256 v2 = _mm256_add_ps(v2a, v2b);

					__m256 s1 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v1, w), _mm256_mul_ps(v2, dw)), _mm256_broadcast_ss(&kernel1[1]));
					__m256 s2 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v2, w), _mm256_mul_ps(v1, dw)), _mm256_broadcast_ss(&kernel1[2]));
					res = _mm256_add_ps(res, _mm256_add_ps(s1, s2));

					// 4, 5

					__m256 v4 = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j + 2)));
					__m256 v5 = _mm256_add_ps(_mm256_loadu_ps((const float*)src.pixeladdr(i + 2, j + 1)), _mm256_loadu_ps((const float*)src.pixeladdr(i + 1, j + 2)));

					__m256 s4 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v4, w), _mm256_mul_ps(v5, dw)), _mm256_broadcast_ss(&kernel1[4]));
					__m256 s5 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v5, w), _mm256_mul_ps(v4, dw)), _mm256_broadcast_ss(&kernel1[5]));
					res = _mm256_add_ps(res, _mm256_add_ps(s4, s5));

					_mm256_storeu_ps((float*)c11.pixeladdr(i + 1, j + 1), res);
				}

				for (int i = Width2; i < src.Width() - 3; i++)
				{
					float w = w1(i, j);
					float dw = 1.0f - w;

					PixelFloatRGBA v0 = src(i, j) + src(i + 3, j + 3);
					PixelFloatRGBA v1 = src(i + 1, j) + src(i, j + 1) + src(i + 3, j + 2) + src(i + 2, j + 3);
					PixelFloatRGBA v2 = src(i + 2, j) + src(i, j + 2) + src(i + 3, j + 1) + src(i + 1, j + 3);
					PixelFloatRGBA v3 = src(i + 3, j) + src(i, j + 3);
					PixelFloatRGBA v4 = src(i + 1, j + 1) + src(i + 2, j + 2);
					PixelFloatRGBA v5 = src(i + 2, j + 1) + src(i + 1, j + 2);

					PixelFloatRGBA res = (v0 * w + v3 * dw) * kernel1[0] +
						(v1 * w + v2 * dw) * kernel1[1] +
						(v2 * w + v1 * dw) * kernel1[2] +
						(v3 * w + v0 * dw) * kernel1[3] +
						(v4 * w + v5 * dw) * kernel1[4] +
						(v5 * w + v4 * dw) * kernel1[5];

					c11(i + 1, j + 1) = res;
				}
			});

			ToGrayScale(c11, r11);
		}

		void DerivativeHorizontal(const Image<float> &src, Image<float> &dst)
		{
			Parallel::For(0, Height, [&src, &dst, this](int j)
			{
				static const __m256 SIGNMASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

				int Width8 = (Width - 1) / 8 + 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 s11 = _mm256_loadu_ps(src.pixeladdr(i + 1, j));
					__m256 s00 = _mm256_load_ps(src.pixeladdr(i, j));
					__m256 diff = _mm256_sub_ps(s11, s00);
					__m256 res = _mm256_and_ps(diff, SIGNMASK);
					_mm256_store_ps(dst.pixeladdr(i, j), res);
				}

				for (int i = 0; i < Width - 1; i++)
					dst(i, j) = fabsf(src(i + 1, j) - src(i, j));

				dst(Width - 1, j) = 0.0f;
			});
		}

		void DerivativeVertical(const Image<float> &src, Image<float> &dst)
		{
			Parallel::For(0, Height - 1, [&src, &dst, this](int j)
			{
				static const __m256 SIGNMASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

				int Width8 = Width / 8 + 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 s11 = _mm256_load_ps(src.pixeladdr(i, j + 1));
					__m256 s00 = _mm256_load_ps(src.pixeladdr(i, j));
					__m256 diff = _mm256_sub_ps(s11, s00);
					__m256 res = _mm256_and_ps(diff, SIGNMASK);
					_mm256_store_ps(dst.pixeladdr(i, j), res);
				}

				for (int i = 0; i < Width; i++)
					dst(i, j) = fabsf(src(i, j + 1) - src(i, j));
			});

			for (int i = 0; i < Width; i++)
				dst(i, Height - 1) = 0.0f;
		}

		void UseDerivativeHorizontal00()
		{
			Parallel::For(0, Height - 2, [this](int j)
			{
				int Width8 = (Width - 2) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 v1 = _mm256_add_ps(_mm256_loadu_ps(tmp.pixeladdr(i + 1, j)), _mm256_loadu_ps(tmp.pixeladdr(i, j + 1)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 1)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 2)));
					_mm256_storeu_ps(pw1.pixeladdr(i + 1, j + 1), v1);

					__m256 v2 = _mm256_add_ps(_mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 2)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 2)));
					_mm256_storeu_ps(pw2.pixeladdr(i + 2, j + 1), v2);
				}

				for (int i = Width8; i < Width - 2; i++)
				{
					pw1(i + 1, j + 1) = tmp(i + 1, j) + tmp(i, j + 1) + tmp(i + 1, j + 1) + tmp(i + 2, j + 1) + tmp(i + 1, j + 2);
					pw2(i + 2, j + 1) = tmp(i + 1, j + 1) + tmp(i + 2, j + 1) + tmp(i + 1, j + 2) + tmp(i + 2, j + 2);
				}
			});
		}

		void UseDerivativeHorizontal11()
		{
			Parallel::For(0, Height - 2, [this](int j)
			{
				int Width8 = (Width - 2) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 v1 = _mm256_loadu_ps(pw1.pixeladdr(i + 1, j + 1));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i, j)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i, j + 1)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)));
					_mm256_storeu_ps(pw1.pixeladdr(i + 1, j + 1), v1);

					__m256 v2 = _mm256_loadu_ps(pw2.pixeladdr(i + 2, j + 1));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 2)));
					_mm256_storeu_ps(pw2.pixeladdr(i + 2, j + 1), v2);
				}

				for (int i = Width8; i < Width - 2; i++)
				{
					pw1(i + 1, j + 1) += tmp(i, j) + tmp(i + 1, j) + tmp(i, j + 1) + tmp(i + 1, j + 1);
					pw2(i + 2, j + 1) += tmp(i + 1, j) + tmp(i, j + 1) + tmp(i + 1, j + 1) + tmp(i + 2, j + 1) + tmp(i + 1, j + 2);
				}
			});
		}

		void UseDerivativeVertical00()
		{
			Parallel::For(0, Height - 2, [this](int j)
			{
				int Width8 = (Width - 2) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 v1 = _mm256_add_ps(_mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 1)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 2)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 2)));
					_mm256_storeu_ps(qw1.pixeladdr(i + 1, j + 2), v1);

					__m256 v2 = _mm256_add_ps(_mm256_loadu_ps(tmp.pixeladdr(i + 1, j)), _mm256_loadu_ps(tmp.pixeladdr(i, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 2)));
					_mm256_storeu_ps(qw2.pixeladdr(i + 1, j + 1), v2);
				}

				for (int i = Width8; i < Width - 2; i++)
				{
					qw1(i + 1, j + 2) = tmp(i + 1, j + 1) + tmp(i + 2, j + 1) + tmp(i + 1, j + 2) + tmp(i + 2, j + 2);
					qw2(i + 1, j + 1) = tmp(i + 1, j) + tmp(i, j + 1) + tmp(i + 1, j + 1) + tmp(i + 2, j + 1) + tmp(i + 1, j + 2);
				}
			});
		}

		void UseDerivativeVertical11()
		{
			Parallel::For(0, Height - 2, [this](int j)
			{
				int Width8 = (Width - 2) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 v1 = _mm256_loadu_ps(qw1.pixeladdr(i + 1, j + 2));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i, j + 1)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 2, j + 2)));
					v1 = _mm256_add_ps(v1, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 2)));
					_mm256_storeu_ps(qw1.pixeladdr(i + 1, j + 2), v1);

					__m256 v2 = _mm256_loadu_ps(qw2.pixeladdr(i + 1, j + 1));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i, j)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i, j + 1)));
					v2 = _mm256_add_ps(v2, _mm256_loadu_ps(tmp.pixeladdr(i + 1, j + 1)));
					_mm256_storeu_ps(qw2.pixeladdr(i + 1, j + 1), v2);
				}

				for (int i = Width8; i < Width - 2; i++)
				{
					qw1(i + 1, j + 2) += tmp(i + 1, j) + tmp(i, j + 1) + tmp(i + 1, j + 1) + tmp(i + 2, j + 1) + tmp(i + 1, j + 2);
					qw2(i + 1, j + 1) += tmp(i, j) + tmp(i + 1, j) + tmp(i, j + 1) + tmp(i + 1, j + 1);
				}
			});
		}

		void ToWeights2()
		{
			Parallel::For(2, Height - 1, [this](int j)
			{
				int Width8 = (Width - 2) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					_mm256_store_ps(w1.pixeladdr(i + 1, j), CalcWeightsFast(_mm256_load_ps(pw1.pixeladdr(i + 1, j)), _mm256_load_ps(qw1.pixeladdr(i + 1, j))));
				}

				for (int i = Width8; i < Width - 2; i++)
				{
					w1(i + 1, j) = CalcWeightsFast(pw1(i + 1, j), qw1(i + 1, j));
					// float p6 = pow6(pw1(i + 1, j)), q6 = pow6(qw1(i + 1, j));
					// w1(i + 1, j) = (1.0f + p6) / (2.0f + p6 + q6);
				}
			});

			Parallel::For(1, Height - 1, [this](int j)
			{
				int Width8 = (Width - 3) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					_mm256_store_ps(w2.pixeladdr(i + 2, j), CalcWeightsFast(_mm256_load_ps(pw2.pixeladdr(i + 2, j)), _mm256_load_ps(qw2.pixeladdr(i + 2, j))));
				}

				for (int i = Width8; i < Width - 3; i++)
				{
					w2(i + 2, j) = CalcWeightsFast(pw2(i + 2, j), qw2(i + 2, j));
					// float p6 = pow6(pw2(i + 2, j)), q6 = pow6(qw2(i + 2, j));
					// w2(i + 2, j) = (1.0f + p6) / (2.0f + p6 + q6);
				}
			});
		}

		void CalcWeights2()
		{
			DerivativeHorizontal(r00, tmp);
			UseDerivativeHorizontal00();

			DerivativeHorizontal(r11, tmp);
			UseDerivativeHorizontal11();

			DerivativeVertical(r00, tmp);
			UseDerivativeVertical00();

			DerivativeVertical(r11, tmp);
			UseDerivativeVertical11();

			ToWeights2();
		}

		void Step2()
		{
			Parallel::For(0, Height - 3, [this](int j)
			{
				static const __m256 ONES = _mm256_set1_ps(1.0f);

				int Width8 = (Width - 3) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 w = _mm256_load_ps(w1.pixeladdr(i + 1, j + 2));
					__m256 dw = _mm256_sub_ps(ONES, w);

					// 0, 3

					__m256 v0 = _mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i + 1, j)), _mm256_loadu_ps(r11.pixeladdr(i + 1, j + 3)));
					__m256 v3 = _mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i, j + 2)), _mm256_loadu_ps(r00.pixeladdr(i + 3, j + 2)));

					__m256 s0 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v0, w), _mm256_mul_ps(v3, dw)), _mm256_broadcast_ss(&kernel2[0]));
					__m256 s3 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v3, w), _mm256_mul_ps(v0, dw)), _mm256_broadcast_ss(&kernel2[3]));

					__m256 res = _mm256_add_ps(s0, s3);

					// 1, 2

					__m256 v1 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(r00.pixeladdr(i + 2, j + 1))),
						_mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i + 1, j + 3)), _mm256_loadu_ps(r00.pixeladdr(i + 2, j + 3))));

					__m256 v2 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i, j + 1)), _mm256_loadu_ps(r11.pixeladdr(i + 2, j + 1))),
						_mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i, j + 2)), _mm256_loadu_ps(r11.pixeladdr(i + 2, j + 2))));

					__m256 s1 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v1, w), _mm256_mul_ps(v2, dw)), _mm256_broadcast_ss(&kernel2[1]));
					__m256 s2 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v2, w), _mm256_mul_ps(v1, dw)), _mm256_broadcast_ss(&kernel2[2]));

					res = _mm256_add_ps(res, _mm256_add_ps(s1, s2));

					// 4, 5

					__m256 v4 = _mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(r11.pixeladdr(i + 1, j + 2)));
					__m256 v5 = _mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i + 1, j + 2)), _mm256_loadu_ps(r00.pixeladdr(i + 2, j + 2)));

					__m256 s4 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v4, w), _mm256_mul_ps(v5, dw)), _mm256_broadcast_ss(&kernel2[4]));
					__m256 s5 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v5, w), _mm256_mul_ps(v4, dw)), _mm256_broadcast_ss(&kernel2[5]));

					res = _mm256_add_ps(res, _mm256_add_ps(s4, s5));

					_mm256_storeu_ps(r10.pixeladdr(i + 1, j + 2), res);
				}

				for (int i = Width8; i < Width - 3; i++)
				{
					float w = w1(i + 1, j + 2), dw = 1.0f - w;

					float v0 = r11(i + 1, j) + r11(i + 1, j + 3);
					float v1 = r00(i + 1, j + 1) + r00(i + 2, j + 1) + r00(i + 1, j + 3) + r00(i + 2, j + 3);
					float v2 = r11(i, j + 1) + r11(i + 2, j + 1) + r11(i, j + 2) + r11(i + 2, j + 2);
					float v3 = r00(i, j + 2) + r00(i + 3, j + 2);
					float v4 = r11(i + 1, j + 1) + r11(i + 1, j + 2);
					float v5 = r00(i + 1, j + 2) + r00(i + 2, j + 2);

					float res = (v0 * w + v3 * dw) * kernel2[0] +
						(v1 * w + v2 * dw) * kernel2[1] +
						(v2 * w + v1 * dw) * kernel2[2] +
						(v3 * w + v0 * dw) * kernel2[3] +
						(v4 * w + v5 * dw) * kernel2[4] +
						(v5 * w + v4 * dw) * kernel2[5];

					r10(i + 1, j + 2) = res;
				}
			});

			Parallel::For(0, Height - 3, [this](int j)
			{
				static const __m256 ONES = _mm256_set1_ps(1.0f);

				int Width8 = (Width - 3) / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 w = _mm256_load_ps(w2.pixeladdr(i + 2, j + 1));
					__m256 dw = _mm256_sub_ps(ONES, w);

					// 0, 3

					__m256 v0 = _mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i + 2, j)), _mm256_loadu_ps(r00.pixeladdr(i + 2, j + 3)));
					__m256 v3 = _mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i, j + 1)), _mm256_loadu_ps(r11.pixeladdr(i + 3, j + 1)));

					__m256 s0 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v0, w), _mm256_mul_ps(v3, dw)), _mm256_broadcast_ss(&kernel2[0]));
					__m256 s3 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v3, w), _mm256_mul_ps(v0, dw)), _mm256_broadcast_ss(&kernel2[3]));

					__m256 res = _mm256_add_ps(s0, s3);

					// 1, 2

					__m256 v1 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i + 1, j)), _mm256_loadu_ps(r11.pixeladdr(i + 2, j))),
						_mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i + 1, j + 2)), _mm256_loadu_ps(r11.pixeladdr(i + 2, j + 2))));

					__m256 v2 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(r00.pixeladdr(i + 3, j + 1))),
						_mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i + 1, j + 2)), _mm256_loadu_ps(r00.pixeladdr(i + 3, j + 2))));

					__m256 s1 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v1, w), _mm256_mul_ps(v2, dw)), _mm256_broadcast_ss(&kernel2[1]));
					__m256 s2 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v2, w), _mm256_mul_ps(v1, dw)), _mm256_broadcast_ss(&kernel2[2]));

					res = _mm256_add_ps(res, _mm256_add_ps(s1, s2));

					// 4, 5

					__m256 v4 = _mm256_add_ps(_mm256_loadu_ps(r00.pixeladdr(i + 2, j + 1)), _mm256_loadu_ps(r00.pixeladdr(i + 2, j + 2)));
					__m256 v5 = _mm256_add_ps(_mm256_loadu_ps(r11.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps(r11.pixeladdr(i + 2, j + 1)));

					__m256 s4 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v4, w), _mm256_mul_ps(v5, dw)), _mm256_broadcast_ss(&kernel2[4]));
					__m256 s5 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v5, w), _mm256_mul_ps(v4, dw)), _mm256_broadcast_ss(&kernel2[5]));

					res = _mm256_add_ps(res, _mm256_add_ps(s4, s5));

					_mm256_storeu_ps(r01.pixeladdr(i + 2, j + 1), res);
				}

				for (int i = Width8; i < Width - 3; i++)
				{
					float w = w2(i + 2, j + 1), dw = 1.0f - w;

					float v0 = r00(i + 2, j) + r00(i + 2, j + 3);
					float v1 = r11(i + 1, j) + r11(i + 2, j) + r11(i + 1, j + 2) + r11(i + 2, j + 2);
					float v2 = r00(i + 1, j + 1) + r00(i + 3, j + 1) + r00(i + 1, j + 2) + r00(i + 3, j + 2);
					float v3 = r11(i, j + 1) + r11(i + 3, j + 1);
					float v4 = r00(i + 2, j + 1) + r00(i + 2, j + 2);
					float v5 = r11(i + 1, j + 1) + r11(i + 2, j + 1);

					float res = (v0 * w + v3 * dw) * kernel2[0] +
						(v1 * w + v2 * dw) * kernel2[1] +
						(v2 * w + v1 * dw) * kernel2[2] +
						(v3 * w + v0 * dw) * kernel2[3] +
						(v4 * w + v5 * dw) * kernel2[4] +
						(v5 * w + v4 * dw) * kernel2[5];

					r01(i + 2, j + 1) = res;
				}
			});
		}

		void Step2c()
		{
			Parallel::For(0, Height - 3, [this](int j)
			{
				static const __m256 ONES = _mm256_set1_ps(1.0f);

				int Width8 = (Width - 3) / 2 * 2;

				for (int i = 0; i < Width8; i += 2)
				{
					__m256 w = _mm256_insertf128_ps(_mm256_broadcast_ss(w1.pixeladdr(i + 1, j + 2)), _mm_broadcast_ss(w1.pixeladdr(i + 2, j + 2)), 1);
					__m256 dw = _mm256_sub_ps(ONES, w);

					// 0, 3

					__m256 v0 = _mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i + 1, j)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 1, j + 3)));
					__m256 v3 = _mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i, j + 2)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 3, j + 2)));

					__m256 s0 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v0, w), _mm256_mul_ps(v3, dw)), _mm256_broadcast_ss(&kernel2[0]));
					__m256 s3 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v3, w), _mm256_mul_ps(v0, dw)), _mm256_broadcast_ss(&kernel2[3]));

					__m256 res = _mm256_add_ps(s0, s3);

					// 1, 2

					__m256 v1 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 2, j + 1))),
						_mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i + 1, j + 3)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 2, j + 3))));

					__m256 v2 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i, j + 1)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 2, j + 1))),
						_mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i, j + 2)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 2, j + 2))));

					__m256 s1 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v1, w), _mm256_mul_ps(v2, dw)), _mm256_broadcast_ss(&kernel2[1]));
					__m256 s2 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v2, w), _mm256_mul_ps(v1, dw)), _mm256_broadcast_ss(&kernel2[2]));

					res = _mm256_add_ps(res, _mm256_add_ps(s1, s2));

					// 4, 5

					__m256 v4 = _mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 1, j + 2)));
					__m256 v5 = _mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i + 1, j + 2)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 2, j + 2)));

					__m256 s4 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v4, w), _mm256_mul_ps(v5, dw)), _mm256_broadcast_ss(&kernel2[4]));
					__m256 s5 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v5, w), _mm256_mul_ps(v4, dw)), _mm256_broadcast_ss(&kernel2[5]));

					res = _mm256_add_ps(res, _mm256_add_ps(s4, s5));

					_mm256_storeu_ps((float*)c10.pixeladdr(i + 1, j + 2), res);
				}

				for (int i = Width8; i < Width - 3; i++)
				{
					float w = w1(i + 1, j + 2), dw = 1.0f - w;

					PixelFloatRGBA v0 = c11(i + 1, j) + c11(i + 1, j + 3);
					PixelFloatRGBA v1 = c00(i + 1, j + 1) + c00(i + 2, j + 1) + c00(i + 1, j + 3) + c00(i + 2, j + 3);
					PixelFloatRGBA v2 = c11(i, j + 1) + c11(i + 2, j + 1) + c11(i, j + 2) + c11(i + 2, j + 2);
					PixelFloatRGBA v3 = c00(i, j + 2) + c00(i + 3, j + 2);
					PixelFloatRGBA v4 = c11(i + 1, j + 1) + c11(i + 1, j + 2);
					PixelFloatRGBA v5 = c00(i + 1, j + 2) + c00(i + 2, j + 2);

					PixelFloatRGBA res = (v0 * w + v3 * dw) * kernel2[0] +
						(v1 * w + v2 * dw) * kernel2[1] +
						(v2 * w + v1 * dw) * kernel2[2] +
						(v3 * w + v0 * dw) * kernel2[3] +
						(v4 * w + v5 * dw) * kernel2[4] +
						(v5 * w + v4 * dw) * kernel2[5];

					c10(i + 1, j + 2) = res;
				}
			});

			Parallel::For(0, Height - 3, [this](int j)
			{
				static const __m256 ONES = _mm256_set1_ps(1.0f);

				int Width8 = (Width - 3) / 2 * 2;

				for (int i = 0; i < Width8; i += 2)
				{
					__m256 w = _mm256_insertf128_ps(_mm256_broadcast_ss(w2.pixeladdr(i + 2, j + 1)), _mm_broadcast_ss(w2.pixeladdr(i + 3, j + 1)), 1);
					__m256 dw = _mm256_sub_ps(ONES, w);

					// 0, 3

					__m256 v0 = _mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i + 2, j)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 2, j + 3)));
					__m256 v3 = _mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i, j + 1)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 3, j + 1)));

					__m256 s0 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v0, w), _mm256_mul_ps(v3, dw)), _mm256_broadcast_ss(&kernel2[0]));
					__m256 s3 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v3, w), _mm256_mul_ps(v0, dw)), _mm256_broadcast_ss(&kernel2[3]));

					__m256 res = _mm256_add_ps(s0, s3);

					// 1, 2

					__m256 v1 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i + 1, j)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 2, j))),
						_mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i + 1, j + 2)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 2, j + 2))));

					__m256 v2 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 3, j + 1))),
						_mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i + 1, j + 2)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 3, j + 2))));

					__m256 s1 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v1, w), _mm256_mul_ps(v2, dw)), _mm256_broadcast_ss(&kernel2[1]));
					__m256 s2 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v2, w), _mm256_mul_ps(v1, dw)), _mm256_broadcast_ss(&kernel2[2]));

					res = _mm256_add_ps(res, _mm256_add_ps(s1, s2));

					// 4, 5

					__m256 v4 = _mm256_add_ps(_mm256_loadu_ps((const float*)c00.pixeladdr(i + 2, j + 1)), _mm256_loadu_ps((const float*)c00.pixeladdr(i + 2, j + 2)));
					__m256 v5 = _mm256_add_ps(_mm256_loadu_ps((const float*)c11.pixeladdr(i + 1, j + 1)), _mm256_loadu_ps((const float*)c11.pixeladdr(i + 2, j + 1)));

					__m256 s4 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v4, w), _mm256_mul_ps(v5, dw)), _mm256_broadcast_ss(&kernel2[4]));
					__m256 s5 = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(v5, w), _mm256_mul_ps(v4, dw)), _mm256_broadcast_ss(&kernel2[5]));

					res = _mm256_add_ps(res, _mm256_add_ps(s4, s5));

					_mm256_storeu_ps((float*)c01.pixeladdr(i + 2, j + 1), res);

				}

				for (int i = Width8; i < Width - 3; i++)
				{
					float w = w2(i + 2, j + 1), dw = 1.0f - w;

					PixelFloatRGBA v0 = c00(i + 2, j) + c00(i + 2, j + 3);
					PixelFloatRGBA v1 = c11(i + 1, j) + c11(i + 2, j) + c11(i + 1, j + 2) + c11(i + 2, j + 2);
					PixelFloatRGBA v2 = c00(i + 1, j + 1) + c00(i + 3, j + 1) + c00(i + 1, j + 2) + c00(i + 3, j + 2);
					PixelFloatRGBA v3 = c11(i, j + 1) + c11(i + 3, j + 1);
					PixelFloatRGBA v4 = c00(i + 2, j + 1) + c00(i + 2, j + 2);
					PixelFloatRGBA v5 = c11(i + 1, j + 1) + c11(i + 2, j + 1);

					PixelFloatRGBA res = (v0 * w + v3 * dw) * kernel2[0] +
						(v1 * w + v2 * dw) * kernel2[1] +
						(v2 * w + v1 * dw) * kernel2[2] +
						(v3 * w + v0 * dw) * kernel2[3] +
						(v4 * w + v5 * dw) * kernel2[4] +
						(v5 * w + v4 * dw) * kernel2[5];

					c01(i + 2, j + 1) = res;
				}
			});
		}

		void MakeResult(Image<float> &dst)
		{
			Parallel::For(0, Height, [this, &dst](int j)
			{
				int Width8 = Width / 8 * 8;

				for (int i = 0; i < Width8; i += 8)
				{
					__m256 v00 = _mm256_load_ps(r00.pixeladdr(i, j));
					__m256 v10 = _mm256_load_ps(r10.pixeladdr(i, j));
					__m256 v01 = _mm256_load_ps(r01.pixeladdr(i, j));
					__m256 v11 = _mm256_load_ps(r11.pixeladdr(i, j));

					__m256 p0 = _mm256_unpacklo_ps(v00, v10);
					__m256 p1 = _mm256_unpackhi_ps(v00, v10);

					_mm256_store_ps(dst.pixeladdr(2 * i, 2 * j), _mm256_permute2f128_ps(p0, p1, 0x20));
					_mm256_store_ps(dst.pixeladdr(2 * i + 8, 2 * j), _mm256_permute2f128_ps(p0, p1, 0x31));

					__m256 q0 = _mm256_unpacklo_ps(v01, v11);
					__m256 q1 = _mm256_unpackhi_ps(v01, v11);

					_mm256_store_ps(dst.pixeladdr(2 * i, 2 * j + 1), _mm256_permute2f128_ps(q0, q1, 0x20));
					_mm256_store_ps(dst.pixeladdr(2 * i + 8, 2 * j + 1), _mm256_permute2f128_ps(q0, q1, 0x31));

				}

				for (int i = Width8; i < Width; i++)
				{
					dst(2 * i, 2 * j) = r00(i, j);
					dst(2 * i + 1, 2 * j) = r10(i, j);
					dst(2 * i, 2 * j + 1) = r01(i, j);
					dst(2 * i + 1, 2 * j + 1) = r11(i, j);
				}
			});
		}

		void MakeResult(ImageFloatColor &dst)
		{
			Parallel::For(0, Height, [this, &dst](int j)
			{
				int Width8 = Width / 2 * 2;

				for (int i = 0; i < Width8; i += 2)
				{
					__m256 v00 = _mm256_load_ps((const float*)c00.pixeladdr(i, j));
					__m256 v10 = _mm256_load_ps((const float*)c10.pixeladdr(i, j));
					__m256 v01 = _mm256_load_ps((const float*)c01.pixeladdr(i, j));
					__m256 v11 = _mm256_load_ps((const float*)c11.pixeladdr(i, j));

					_mm256_store_ps((float*)dst.pixeladdr(2 * i, 2 * j), _mm256_permute2f128_ps(v00, v10, 0x20));
					_mm256_store_ps((float*)dst.pixeladdr(2 * i + 2, 2 * j), _mm256_permute2f128_ps(v00, v10, 0x31));

					_mm256_store_ps((float*)dst.pixeladdr(2 * i, 2 * j + 1), _mm256_permute2f128_ps(v01, v11, 0x20));
					_mm256_store_ps((float*)dst.pixeladdr(2 * i + 2, 2 * j + 1), _mm256_permute2f128_ps(v01, v11, 0x31));

				}

				for (int i = Width8; i < Width; i++)
				{
					dst(2 * i, 2 * j) = c00(i, j);
					dst(2 * i + 1, 2 * j) = c10(i, j);
					dst(2 * i, 2 * j + 1) = c01(i, j);
					dst(2 * i + 1, 2 * j + 1) = c11(i, j);
				}
			});
		}

		void InitBorders(const Image<float> &src)
		{
			// First 3 rows

			for (int i = 0; i < Width - 1; i++)
			{
				r00(i, 0) = src(i, 0);
				r00(i, 1) = src(i, 1);
				r10(i, 0) = (src(i, 0) + src(i + 1, 0)) * 0.5f;
				r10(i, 1) = (src(i, 1) + src(i + 1, 1)) * 0.5f;
				r01(i, 0) = (src(i, 0) + src(i, 1)) * 0.5f;
				r11(i, 0) = (r10(i, 0) + r10(i, 1)) * 0.5f;
			}

			r00(Width - 1, 0) = src(Width - 1, 0);
			r00(Width - 1, 1) = src(Width - 1, 1);
			r10(Width - 1, 0) = src(Width - 1, 0);
			r10(Width - 1, 1) = src(Width - 1, 1);
			r01(Width - 1, 0) = r11(Width - 1, 0) = (src(Width - 1, 0) + src(Width - 1, 1)) * 0.5f;

			// Last 3 rows

			for (int i = 0; i < Width - 1; i++)
			{
				r00(i, Height - 2) = src(i, Height - 2);
				r00(i, Height - 1) = src(i, Height - 1);
				r10(i, Height - 2) = (src(i, Height - 2) + src(i + 1, Height - 2)) * 0.5f;
				r10(i, Height - 1) = r11(i, Height - 1) = (src(i, Height - 1) + src(i + 1, Height - 1)) * 0.5f;
				r01(i, Height - 2) = (src(i, Height - 2) + src(i, Height - 1)) * 0.5f;
				r01(i, Height - 1) = src(i, Height - 1);
				r11(i, Height - 2) = (r10(i, Height - 2) + r10(i, Height - 1)) * 0.5f;
			}

			r00(Width - 1, Height - 2) = src(Width - 1, Height - 2);
			r00(Width - 1, Height - 1) = src(Width - 1, Height - 1);
			r10(Width - 1, Height - 2) = src(Width - 1, Height - 2);
			r10(Width - 1, Height - 1) = src(Width - 1, Height - 1);
			r01(Width - 1, Height - 2) = r11(Width - 1, Height - 2) = (src(Width - 1, Height - 2) + src(Width - 1, Height - 1)) * 0.5f;
			r01(Width - 1, Height - 1) = r11(Width - 1, Height - 1) = src(Width - 1, Height - 1);

			// Left and right

			for (int j = 1; j < Height - 1; j++)
			{
				r00(0, j) = src(0, j);
				r00(1, j) = src(1, j);
				r00(Width - 2, j) = src(Width - 2, j);
				r00(Width - 1, j) = src(Width - 1, j);

				r01(0, j) = (src(0, j) + src(0, j + 1)) * 0.5f;
				r01(1, j) = (src(1, j) + src(1, j + 1)) * 0.5f;
				r01(Width - 2, j) = (src(Width - 2, j) + src(Width - 2, j + 1)) * 0.5f;
				r01(Width - 1, j) = (src(Width - 1, j) + src(Width - 1, j + 1)) * 0.5f;

				r10(0, j) = (src(0, j) + src(1, j)) * 0.5f;
				r10(Width - 2, j) = (src(Width - 2, j) + src(Width - 11, j)) * 0.5f;
				r10(Width - 1, j) = src(Width - 1, j);

				r11(0, j) = (r01(0, j) + r01(1, j)) * 0.5f;
				r11(Width - 2, j) = (r01(Width - 2, j) + r10(Width - 1, j)) * 0.5f;
				r11(Width - 1, j) = r01(Width - 1, j);
			}
		}

		void InitBorders(const ImageFloatColor &src)
		{
			// First 3 rows

			for (int i = 0; i < Width - 1; i++)
			{
				c00(i, 0) = src(i, 0);
				c00(i, 1) = src(i, 1);
				c10(i, 0) = (src(i, 0) + src(i + 1, 0)) * 0.5f;
				c10(i, 1) = (src(i, 1) + src(i + 1, 1)) * 0.5f;
				c01(i, 0) = (src(i, 0) + src(i, 1)) * 0.5f;
				c11(i, 0) = (c10(i, 0) + c10(i, 1)) * 0.5f;
			}

			c00(Width - 1, 0) = src(Width - 1, 0);
			c00(Width - 1, 1) = src(Width - 1, 1);
			c10(Width - 1, 0) = src(Width - 1, 0);
			c10(Width - 1, 1) = src(Width - 1, 1);
			c01(Width - 1, 0) = c11(Width - 1, 0) = (src(Width - 1, 0) + src(Width - 1, 1)) * 0.5f;

			// Last 3 rows

			for (int i = 0; i < Width - 1; i++)
			{
				c00(i, Height - 2) = src(i, Height - 2);
				c00(i, Height - 1) = src(i, Height - 1);
				c10(i, Height - 2) = (src(i, Height - 2) + src(i + 1, Height - 2)) * 0.5f;
				c10(i, Height - 1) = c11(i, Height - 1) = (src(i, Height - 1) + src(i + 1, Height - 1)) * 0.5f;
				c01(i, Height - 2) = (src(i, Height - 2) + src(i, Height - 1)) * 0.5f;
				c01(i, Height - 1) = src(i, Height - 1);
				c11(i, Height - 2) = (c10(i, Height - 2) + c10(i, Height - 1)) * 0.5f;
			}

			c00(Width - 1, Height - 2) = src(Width - 1, Height - 2);
			c00(Width - 1, Height - 1) = src(Width - 1, Height - 1);
			c10(Width - 1, Height - 2) = src(Width - 1, Height - 2);
			c10(Width - 1, Height - 1) = src(Width - 1, Height - 1);
			c01(Width - 1, Height - 2) = c11(Width - 1, Height - 2) = (src(Width - 1, Height - 2) + src(Width - 1, Height - 1)) * 0.5f;
			c01(Width - 1, Height - 1) = c11(Width - 1, Height - 1) = src(Width - 1, Height - 1);

			// Left and right

			for (int j = 1; j < Height - 1; j++)
			{
				c00(0, j) = src(0, j);
				c00(1, j) = src(1, j);
				c00(Width - 2, j) = src(Width - 2, j);
				c00(Width - 1, j) = src(Width - 1, j);

				c01(0, j) = (src(0, j) + src(0, j + 1)) * 0.5f;
				c01(1, j) = (src(1, j) + src(1, j + 1)) * 0.5f;
				c01(Width - 2, j) = (src(Width - 2, j) + src(Width - 2, j + 1)) * 0.5f;
				c01(Width - 1, j) = (src(Width - 1, j) + src(Width - 1, j + 1)) * 0.5f;

				c10(0, j) = (src(0, j) + src(1, j)) * 0.5f;
				c10(Width - 2, j) = (src(Width - 2, j) + src(Width - 11, j)) * 0.5f;
				c10(Width - 1, j) = src(Width - 1, j);

				c11(0, j) = (c01(0, j) + c01(1, j)) * 0.5f;
				c11(Width - 2, j) = (c01(Width - 2, j) + c10(Width - 1, j)) * 0.5f;
				c11(Width - 1, j) = c01(Width - 1, j);
			}
		}

		float& pixel (int x, int y)
		{
			if ((y % 2) == 0)
				if ((x % 2) == 0)
					return r00(x / 2, y / 2);
				else
					return r10(x / 2, y / 2);
			else
				if ((x % 2) == 0)
					return r01(x / 2, y / 2);
				else
					return r11(x / 2, y / 2);
		}

		void Step2FailSafe()
		{
			for (int j = 3; j < Height * 2 - 3; j++)
			{
				for (int i = 3 + (j % 2); i < Width * 2 - 3; i += 2)
				{
					float p = fabsf(pixel(i, j - 1) - pixel(i, j - 3)) +
						fabsf(pixel(i - 1, j) - pixel(i - 1, j - 2)) + fabsf(pixel(i + 1, j) - pixel(i + 1, j - 2)) +
						fabsf(pixel(i - 2, j + 1) - pixel(i - 2, j - 1)) + fabsf(pixel(i, j + 1) - pixel(i, j - 1)) + fabsf(pixel(i + 2, j + 1) - pixel(i + 2, j - 1)) +
						fabsf(pixel(i - 1, j + 2) - pixel(i - 1, j)) + fabsf(pixel(i + 1, j + 2) - pixel(i + 1, j)) +
						fabsf(pixel(i, j + 3) - pixel(i, j + 1));

					float q = fabsf(pixel(i + 1, j - 2) - pixel(i - 1, j - 2)) +
						fabsf(pixel(i + 2, j - 1) - pixel(i, j - 1)) + fabsf(pixel(i, j - 1) - pixel(i - 2, j - 1)) +
						fabsf(pixel(i + 3, j) - pixel(i + 1, j)) + fabsf(pixel(i + 1, j) - pixel(i - 1, j)) + fabsf(pixel(i - 1, j) - pixel(i - 3, j)) +
						fabsf(pixel(i + 2, j + 1) - pixel(i, j + 1)) + fabsf(pixel(i, j + 1) - pixel(i - 2, j + 1)) +
						fabsf(pixel(i + 1, j + 2) - pixel(i - 1, j + 2));

					float g1 = pow6(p), g2 = pow6(p);
					float w = (1.0f + g1) / (2.0f + g1 + g2), dw = 1.0f - w;

					float v0 = pixel(i, j - 3) + pixel(i, j + 3);
					float v1 = pixel(i - 1, j - 2) + pixel(i + 1, j - 2) + pixel(i - 1, j + 2) + pixel(i + 1, j + 2);
					float v2 = pixel(i - 2, j - 1) + pixel(i + 2, j - 1) + pixel(i - 2, j + 1) + pixel(i + 2, j + 1);
					float v3 = pixel(i - 3, j) + pixel(i + 3, j);
					float v4 = pixel(i, j - 1) + pixel(i, j + 1);
					float v5 = pixel(i - 1, j) + pixel(i + 1, j);

					float res = (v0 * w + v3 * dw) * kernel2[0] +
						(v1 * w + v2 * dw) * kernel2[1] +
						(v2 * w + v1 * dw) * kernel2[2] +
						(v3 * w + v0 * dw) * kernel2[3] +
						(v4 * w + v5 * dw) * kernel2[4] +
						(v5 * w + v4 * dw) * kernel2[5];

					pixel(i, j) = res;
				}
			}
		}

	public:
		void Perform(const Image<float> &src, Image<float> &dst)
		{
			check(src.Width() == Width && src.Height() == Height);
			check(dst.Width() == Width * 2 && dst.Height() == Height * 2);

			InitBorders(src);

			Step00(src);

			CalcWeights1(src);
			Step11(src);

			CalcWeights2();
			Step2();

			MakeResult(dst);
		}

		void Perform(const ImageFloatColor &src, ImageFloatColor &dst)
		{
			check(src.Width() == Width && src.Height() == Height);
			check(dst.Width() == Width * 2 && dst.Height() == Height * 2);

			InitBorders(src);

			Step00(src);

			CalcWeights1(r00);
			Step11(src);

			CalcWeights2();
			Step2c();

			MakeResult(dst);
		}
	};

	// ==============================================================================

	bool EDR_Resampling_x2(const ip::Image<float> &src, ip::Image<float> &dst)
	{
		EDRFast(src.Width(), src.Height()).Perform(src, dst);
		return true;
	}

	bool EDR_Resampling_x2(const ip::ImageFloatColor &src, ip::ImageFloatColor &dst)
	{
		EDRFast(src.Width(), src.Height()).Perform(src, dst);
		return true;
	}
}

#include "srcnn.h"

#include <fstream>
#include <chrono>
#include <iplib/parallel.h>
#include <iplib/image/core3d.h>
#include <iplib/image/filter.h>

#include <misc/simd.h>

using namespace std;

namespace ip
{
	/* __declspec(align(32)) struct float8
	{
		float p[8];

		inline float8() = default;

		inline float8(float v)
		{
			p[0] = p[1] = p[2] = p[3] = p[4] = p[5] = p[6] = p[7] = v;
		}

		inline float8& operator += (const float8 &a)
		{
			for (int i = 0; i < 8; i++)
				p[i] += a.p[i];
			return *this;
		}

		inline float8 max(const float8 &other) const
		{
			float8 res;
			res.p[0] = p[0] > other.p[0] ? p[0] : other.p[0];
			res.p[1] = p[1] > other.p[1] ? p[1] : other.p[1];
			res.p[2] = p[2] > other.p[2] ? p[2] : other.p[2];
			res.p[3] = p[3] > other.p[3] ? p[3] : other.p[3];
			res.p[4] = p[4] > other.p[4] ? p[4] : other.p[4];
			res.p[5] = p[5] > other.p[5] ? p[5] : other.p[5];
			res.p[6] = p[6] > other.p[6] ? p[6] : other.p[6];
			res.p[7] = p[7] > other.p[7] ? p[7] : other.p[7];
			return res;
		}
	};

	inline float8 operator + (const float8 &a, const float8 &b)
	{
		float8 res;
		for (int i = 0; i < 8; i++)
			res.p[i] = a.p[i] + b.p[i];
		return res;
	}

	inline float8 operator * (const float8 &a, const float8 &b)
	{
		float8 res;
		for (int i = 0; i < 8; i++)
			res.p[i] = a.p[i] * b.p[i];
		return res;
	}

	inline float8 operator * (float a, const float8 &b)
	{
		float8 res;
		for (int i = 0; i < 8; i++)
			res.p[i] = a * b.p[i];
		return res;
	} */

	// ========================================================================

	class SRCNN_Resampling
	{
	private:
		float biases1[64];
		float biases2[32];
		float biases3;

		float weights_conv1[64][9][9];
		float weights_conv2[32][1][1][64];
		float weights_conv3[5][5][32];

		// float weights_conv1[81][64];
		// float weights_conv2[64][25][32];
		// float weights_conv3[32][25];


	public:
		bool LoadData(const wchar_t *filename);

		void BicubicInitialization(const ip::Image<float> &src, ip::Image<float> &dst);
		void BicubicInitialization2(const ip::Image<float> &src, ip::Image<float> &dst);
		void Process(const ip::Image<float> &src, ip::Image<float> &dst, bool half_shift);
		void ProcessDebug(const ip::Image<float> &src, ip::Image<float> &dst);

		// void ProcessLayer1(const Image<float> &src, Image3D<float8> &dst);
		// void ProcessLayer2(const Image3D<float8> &src, Image3D<float8> &dst);
		// void ProcessLayer12_old(const Image<float> &src, Image3D<VectorFloat> &dst);
		void ProcessLayer12(const Image<float> &src, Image3D<VectorFloat> &dst);
		void ProcessLayer3(const Image3D<VectorFloat> &src, Image<float> &dst);
	};

	// =================================================================================================

	bool SRCNN_Resampling::LoadData(const wchar_t *filename)
	{
		fstream fs;

		fs.open(filename, fstream::in | fstream::binary);
		if (fs.fail())
			return false;

		fs.read((char*)&weights_conv1, sizeof(weights_conv1));
		fs.read((char*)&biases1, sizeof(biases1));
		fs.read((char*)&weights_conv2, sizeof(weights_conv2));
		fs.read((char*)&biases2, sizeof(biases2));
		fs.read((char*)&weights_conv3, sizeof(weights_conv3));
		fs.read((char*)&biases3, sizeof(biases3));

		fs.close();
		return true;
	}

	void SRCNN_Resampling::BicubicInitialization(const ip::Image<float> &src, ip::Image<float> &dst)
	{
		check(dst.Width() == src.Width() * 2 && dst.Height() == src.Height() * 2);

		Parallel::For(0, src.Height(), [&src, &dst](int y)
		{
			const float *l0 = src.pixeladdr(0, y);
			const float *l1 = src.pixeladdr(0, y < src.Height() - 1 ? y + 1 : y);

			for (int i = 0; i < src.Width(); i++)
			{
				dst(i * 2, y * 2) = l0[i];
				dst(i * 2, y * 2 + 1) = (l0[i] + l1[i]) * 0.5f;
			}

			for (int j = 0; j < 2; j++)
			{
				int y0 = y * 2 + j;

				dst(1, y0) = (dst(0, y0) + dst(2, y0)) * 0.5f;

				for (int i = 3; i < dst.Width() - 3; i += 2)
				{
					dst(i, y0) = ((dst(i - 1, y0) + dst(i + 1, y0)) * 9.0f - (dst(i - 3, y0) + dst(i + 3, y0))) * (1.0f / 16.0f);
				}

				dst(dst.Width() - 3, y0) = (dst(dst.Width() - 4, y0) + dst(dst.Width() - 2, y0)) * 0.5f;
				dst(dst.Width() - 1, y0) = dst(dst.Width() - 2, y0);
			}
		});
	}

	static float cubic(float f0, float f1, float f2, float f3)
	{
		return (f1 * 111.0f + f2 * 29.0f - f0 * 9.0f - f3 * 3.0f) * (1.0f / 128.0f);
	}

	void SRCNN_Resampling::BicubicInitialization2(const ip::Image<float> &src, ip::Image<float> &dst)
	{
		check(dst.Width() == src.Width() * 2 && dst.Height() == src.Height() * 2);

		Parallel::For(0, src.Height() - 3, [&src, &dst](int y)
		{
			const float *l0 = src.pixeladdr(0, y);
			const float *l1 = src.pixeladdr(0, y + 1);
			const float *l2 = src.pixeladdr(0, y + 2);
			const float *l3 = src.pixeladdr(0, y + 3);

			for (int i = 0; i < src.Width() - 3; i++)
			{
				float v0 = cubic(l0[i], l1[i], l2[i], l3[i]);
				float v1 = cubic(l0[i + 1], l1[i + 1], l2[i + 1], l3[i + 1]);
				float v2 = cubic(l0[i + 2], l1[i + 2], l2[i + 2], l3[i + 2]);
				float v3 = cubic(l0[i + 3], l1[i + 3], l2[i + 3], l3[i + 3]);

				dst(i * 2 + 3, y * 2 + 3) = cubic(v0, v1, v2, v3);
				dst(i * 2 + 4, y * 2 + 3) = cubic(v3, v2, v1, v0);

				float w0 = cubic(l3[i], l2[i], l1[i], l0[i]);
				float w1 = cubic(l3[i + 1], l2[i + 1], l1[i + 1], l0[i + 1]);
				float w2 = cubic(l3[i + 2], l2[i + 2], l1[i + 2], l0[i + 2]);
				float w3 = cubic(l3[i + 3], l2[i + 3], l1[i + 3], l0[i + 3]);

				dst(i * 2 + 3, y * 2 + 4) = cubic(w0, w1, w2, w3);
				dst(i * 2 + 4, y * 2 + 4) = cubic(w3, w2, w1, w0);

				//printf("%f, %f, %f, %f -> %f\n", w0, w1, w2, w3, cubic(w0, w1, w2, w3));
			}
		});
	}

	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType, class DestinationImageType, typename FilterPixelType, class FilterImageType>
	void PFilter2D(const ImageReadable<SourcePixelType, SourceImageType> &src, ImageWritable<DestinationPixelType, DestinationImageType> &dst,
		const ImageReadable<FilterPixelType, FilterImageType> &filter, int filter_center_x, int filter_center_y)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());

		Parallel::For(0, src.Height(), [&src, &filter, &dst, filter_center_x, filter_center_y] (int j)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				DestinationPixelType s = DestinationPixelType();

				for (int y = 0; y < filter.Height(); y++)
					for (int x = 0; x < filter.Width(); x++)
					{
						int xx = i + filter_center_x - x;
						int yy = j + filter_center_y - y;

						s += src(std::max(std::min(xx, src.Width() - 1), 0), std::max(std::min(yy, src.Height() - 1), 0)) * filter(x, y);
					}

				dst(i, j) = s;
			}
		});
	}

	void ProcessBorders(int Width, int Height, int BorderSize, std::function<void(int x, int y)> func)
	{
		for (int j = 0; j < BorderSize; j++)
			for (int i = 0; i < Width; i++)
				func(i, j);

		for (int j = 2; j < Height - BorderSize; j++)
		{
			for (int i = 0; i < BorderSize; i++)
				func(i, j);

			for (int i = Width - BorderSize; i < Width; i++)
				func(i, j);
		}

		for (int j = Height - BorderSize; j < Height; j++)
			for (int i = 0; i < Width; i++)
				func(i, j);
	}

	/* void PFilter2D_AVX_9(const Image<float> &src, Image<float8> &dst, const Image<float8> &filter)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height() && filter.Width() == 9 && filter.Height() == 9);

		ProcessBorders(src.Width(), src.Height(), 4, [&src, &dst, &filter](int x, int y)
		{
			float8 s = float8(0.0f);

			for (int j = 0; j < 9; j++)
				for (int i = 0; i < 9; i++)
				{
					int xx = x + 4 - i;
					int yy = y + 4 - j;

					s += src(std::max(std::min(xx, src.Width() - 1), 0), std::max(std::min(yy, src.Height() - 1), 0)) * filter(i, j);
				}

			dst(x, y) = s;
		});

		Parallel::For(4, src.Height() - 4, [&src, &dst, &filter](int y)
		{
			for (int x = 4; x < src.Width() - 4; x++)
			{
#ifdef LEGACY
				float8 s = float8(0.0f);

				for (int j = 0; j < 9; j++)
					for (int i = 0; i < 9; i++)
					{
						int xx = x + 4 - i;
						int yy = y + 4 - j;

						s += src(xx, yy) * filter(i, j);
					}

				dst(x, y) = s;
#else
				__m256 s = _mm256_setzero_ps();

				for (int j = 0; j < 9; j++)
					for (int i = 0; i < 9; i++)
					{
						int xx = x + 4 - i;
						int yy = y + 4 - j;

						s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_broadcast_ss(src.pixeladdr(xx, yy)), _mm256_load_ps((const float*)filter.pixeladdr(i, j))));
					}

				_mm256_store_ps((float*)dst.pixeladdr(x, y), s);
#endif
			}
		});
	} */

	/* void PFilter2D_AVX_5(const Image<float8> &src, Image<float8> &dst, const Image<float8> &filter)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height() && filter.Width() == 5 && filter.Height() == 5);

		ProcessBorders(src.Width(), src.Height(), 2, [&src, &dst, &filter](int x, int y)
		{
			float8 s = float8();

			for (int j = 0; j < 5; j++)
				for (int i = 0; i < 5; i++)
				{
					int xx = x + 2 - i;
					int yy = y + 2 - j;

					s += src(std::max(std::min(xx, src.Width() - 1), 0), std::max(std::min(yy, src.Height() - 1), 0)) * filter(i, j);
				}

			dst(x, y) = s;
		});

		Parallel::For(2, src.Height() - 2, [&src, &dst, &filter](int y)
		{
			for (int x = 2; x < src.Width() - 2; x++)
			{
#ifdef LEGACY
				float8 s = float8();

				for (int j = 0; j < 5; j++)
					for (int i = 0; i < 5; i++)
					{
						int xx = x + 2 - i;
						int yy = y + 2 - j;

						s += src(xx, yy) * filter(i, j);
					}

				dst(x, y) = s;
#else
				__m256 s = _mm256_setzero_ps();

				for (int j = 0; j < 5; j++)
					for (int i = 0; i < 5; i++)
					{
						int xx = x + 2 - i;
						int yy = y + 2 - j;

						s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_load_ps((const float*)src.pixeladdr(xx, yy)), _mm256_load_ps((const float*)filter.pixeladdr(i, j))));
					}

				_mm256_store_ps((float*)dst.pixeladdr(x, y), s);
#endif
			}
		});
	} */

	/* void PFilter2D_AVX_5_ADD(const Image<float8> &src, Image<float8> &dst, const Image<float8> &filter)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height() && filter.Width() == 5 && filter.Height() == 5);

		ProcessBorders(src.Width(), src.Height(), 2, [&src, &dst, &filter](int x, int y)
		{
			float8 s = dst(x, y);

			for (int j = 0; j < 5; j++)
				for (int i = 0; i < 5; i++)
				{
					int xx = x + 2 - i;
					int yy = y + 2 - j;

					s += src(std::max(std::min(xx, src.Width() - 1), 0), std::max(std::min(yy, src.Height() - 1), 0)) * filter(i, j);
				}

			dst(x, y) = s;
		});

		Parallel::For(2, src.Height() - 2, [&src, &dst, &filter](int y)
		{
			for (int x = 2; x < src.Width() - 2; x++)
			{
#ifdef LEGACY
				float8 s = dst(x, y);

				for (int j = 0; j < 5; j++)
					for (int i = 0; i < 5; i++)
					{
						int xx = x + 2 - i;
						int yy = y + 2 - j;

						s += src(xx, yy) * filter(i, j);
					}

				dst(x, y) = s;
#else
				__m256 s = _mm256_load_ps((float*)dst.pixeladdr(x, y));

				for (int j = 0; j < 5; j++)
					for (int i = 0; i < 5; i++)
					{
						int xx = x + 2 - i;
						int yy = y + 2 - j;

						s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_load_ps((const float*)src.pixeladdr(xx, yy)), _mm256_load_ps((const float*)filter.pixeladdr(i, j))));
					}

				_mm256_store_ps((float*)dst.pixeladdr(x, y), s);
#endif
			}
		});
	} */

	/* void PFilter2D_AVX_5(const Image<float> &src, Image<float8> &dst, const Image<float8> &filter)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height() && filter.Width() == 5 && filter.Height() == 5);

		ProcessBorders(src.Width(), src.Height(), 2, [&src, &dst, &filter](int x, int y)
		{
			float8 s = float8();

			for (int j = 0; j < 5; j++)
				for (int i = 0; i < 5; i++)
				{
					int xx = x + 2 - i;
					int yy = y + 2 - j;

					s += src(std::max(std::min(xx, src.Width() - 1), 0), std::max(std::min(yy, src.Height() - 1), 0)) * filter(i, j);
				}

			dst(x, y) = s;
		});

		Parallel::For(2, src.Height() - 2, [&src, &dst, &filter](int y)
		{
			for (int x = 2; x < src.Width() - 2; x++)
			{
#ifdef LEGACY
				float8 s = float8();

				for (int j = 0; j < 5; j++)
					for (int i = 0; i < 5; i++)
					{
						int xx = x + 2 - i;
						int yy = y + 2 - j;

						s += src(xx, yy) * filter(i, j);
					}

				dst(x, y) = s;
#else
				__m256 s = _mm256_setzero_ps();

				for (int j = 0; j < 5; j++)
					for (int i = 0; i < 5; i++)
					{
						int xx = x + 2 - i;
						int yy = y + 2 - j;

						s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_broadcast_ss(src.pixeladdr(xx, yy)), _mm256_load_ps((const float*)filter.pixeladdr(i, j))));
					}

				_mm256_store_ps((float*)dst.pixeladdr(x, y), s);
#endif
			}
		});
	} */

	/* void SRCNN_Resampling::ProcessLayer1(const Image<float> &src, Image3D<VectorFloat> &dst)
	{
		Image<float8> filter(81, 8);
		float8 bias[8];

		for (int z = 0; z < 8; z++)
		{
			for (int j = 0; j < 9; j++)
				for (int i = 0; i < 9; i++)
					for (int k = 0; k < 8; k++)
						filter(j * 9 + i, z).p[k] = weights_conv1[z * 8 + k][i][j];

			for (int k = 0; k < 8; k++)
				bias[z].p[k] = biases1[z * 8 + k];
		}

		Parallel::For(0, src.Height(), [&src, &dst, &filter, &bias](int y)
		{
			__m256 M1_255 = _mm256_set1_ps(1.0f / 255.0f);

			for (int x = 0; x < src.Width(); x++)
			{
				float v[81];

				if (x >= 4 && x < src.Width() - 4 && y >= 4 && y < src.Height() - 4)
				{
					for (int j = 0; j < 9; j++)
						for (int i = 0; i < 9; i++)
							v[j * 9 + i] = src(x + i - 4, y + j - 4);
				}
				else
				{
					for (int j = 0; j < 9; j++)
						for (int i = 0; i < 9; i++)
							v[j * 9 + i] = src(std::max(std::min(x + i - 4, src.Width() - 1), 0), std::max(std::min(y + j - 4, src.Height() - 1), 0));
				}

				for (int z = 0; z < 8; z++)
				{
					__m256 s = _mm256_setzero_ps();

					for (int i = 0; i < 81; i++)
						s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_broadcast_ss(v + i), _mm256_load_ps(filter(i, z).p)));

					__m256 m_bias = _mm256_load_ps(bias[z].p);
					__m256 res = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(s, M1_255), m_bias), _mm256_setzero_ps());
					_mm256_store_ps(dst(x, y, z).p, res);
				}
			}
		});
	}

	void SRCNN_Resampling::ProcessLayer2(const Image3D<float8> &src, Image3D<float8> &dst)
	{
		for (int k = 0; k < 4; k++)
		{
			float8 filter[64];
			for (int z = 0; z < 64; z++)
				for (int q = 0; q < 8; q++)
					filter[z].p[q] = weights_conv2[k * 8 + q][0][0][z];

			float8 bias;
			for (int q = 0; q < 8; q++)
				bias.p[q] = biases2[k * 8 + q];

			Parallel::For(0, src.SizeY(), [&src, &dst, k, &filter, bias](int j)
			{
#ifdef LEGACY
				float8 M0(0.0f);

				for (int i = 0; i < src.SizeX(); i++)
				{
					float8 sum(0.0f);

					for (int z = 0; z < 8; z++)
						for (int q = 0; q < 8; q++)
							sum += src.pixel(i, j, z).p[q] * filter[z * 8 + q];

					dst(i, j, k) = (sum + bias).max(M0);
				}
#else
				__m256 m_bias = _mm256_loadu_ps(bias.p);

				for (int i = 0; i < src.SizeX(); i++)
				{
					__m256 sum = _mm256_setzero_ps();

					for (int z = 0; z < 8; z++)
						for (int q = 0; q < 8; q++)
							sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_broadcast_ss(src.pixel(i, j, z).p + q), _mm256_load_ps(filter[z * 8 + q].p)));

					sum = _mm256_max_ps(_mm256_add_ps(sum, m_bias), _mm256_setzero_ps());
					_mm256_store_ps(dst(i, j, k).p, sum);
				}
#endif
			});
		}
	} */

	/* void SRCNN_Resampling::ProcessLayer12_old(const Image<float> &src, Image3D<float8> &dst)
	{
		Image<float8> filter1(81, 8);
		Image<float8> filter2(4, 64);
		float8 bias1[8];
		float8 bias2[4];

		for (int z = 0; z < 8; z++)
		{
			for (int j = 0; j < 9; j++)
				for (int i = 0; i < 9; i++)
					for (int k = 0; k < 8; k++)
						filter1(j * 9 + i, z).set(k, weights_conv1[z * 8 + k][i][j]);

			for (int k = 0; k < 8; k++)
				bias1[z].set(k, biases1[z * 8 + k]);
		}

		for (int k = 0; k < 4; k++)
		{
			for (int z = 0; z < 64; z++)
				for (int q = 0; q < 8; q++)
					filter2(k, z).set(q, weights_conv2[k * 8 + q][0][0][z]);

			for (int q = 0; q < 8; q++)
				bias2[k].set(q, biases2[k * 8 + q]);
		}

		Parallel::For(0, src.Height(), [&src, &dst, &filter1, &bias1, &filter2, &bias2](int y)
		{
			__m256 M1_255 = _mm256_set1_ps(1.0f / 255.0f);

			for (int x = 0; x < src.Width(); x++)
			{
				// ----- STEP 1 -----

				float v[81];

				if (x >= 4 && x < src.Width() - 4 && y >= 4 && y < src.Height() - 4)
				{
					for (int j = 0; j < 9; j++)
						for (int i = 0; i < 9; i++)
							v[j * 9 + i] = src(x + i - 4, y + j - 4);
				}
				else
				{
					for (int j = 0; j < 9; j++)
						for (int i = 0; i < 9; i++)
							v[j * 9 + i] = src(std::max(std::min(x + i - 4, src.Width() - 1), 0), std::max(std::min(y + j - 4, src.Height() - 1), 0));
				}

				float8 tmp1[8];

				for (int z = 0; z < 8; z++)
				{
					__m256 s = _mm256_setzero_ps();

					const float *sptr = v;
					const float *fptr = (const float*)filter1.pixeladdr(0, z);

					for (int i = 0; i < 81; i++)
					{
						s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_broadcast_ss(sptr), _mm256_load_ps(fptr)));
						sptr++;
						fptr += 8;
					}

					__m256 m_bias = _mm256_load_ps((const float*)(bias1 + z));
					__m256 res = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(s, M1_255), m_bias), _mm256_setzero_ps());
					_mm256_store_ps((float*)(tmp1 + z), res);
				}

				// ----- STEP 2 -----

				__m256 sum0 = _mm256_setzero_ps();
				__m256 sum1 = _mm256_setzero_ps();
				__m256 sum2 = _mm256_setzero_ps();
				__m256 sum3 = _mm256_setzero_ps();

				for (int z = 0; z < 8; z++)
				{
					const float *sptr = (const float*)(tmp1 + z);

					for (int q = 0; q < 8; q++)
					{
						const float *fptr = (const float*)filter2.pixeladdr(0, z * 8 + q);
						__m256 v = _mm256_broadcast_ss(sptr++);
						sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(v, _mm256_load_ps(fptr)));
						sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(v, _mm256_load_ps(fptr + 8)));
						sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(v, _mm256_load_ps(fptr + 16)));
						sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(v, _mm256_load_ps(fptr + 24)));
					}
				}

				float *dptr = (float*)dst.pixeladdr(0, x, y);

				sum0 = _mm256_max_ps(_mm256_add_ps(sum0, _mm256_loadu_ps((const float*)bias2)), _mm256_setzero_ps());
				_mm256_stream_ps(dptr, sum0);

				sum1 = _mm256_max_ps(_mm256_add_ps(sum1, _mm256_loadu_ps((const float*)(bias2 + 1))), _mm256_setzero_ps());
				_mm256_stream_ps(dptr + 8, sum1);

				sum2 = _mm256_max_ps(_mm256_add_ps(sum2, _mm256_loadu_ps((const float*)(bias2 + 2))), _mm256_setzero_ps());
				_mm256_stream_ps(dptr + 16, sum2);

				sum3 = _mm256_max_ps(_mm256_add_ps(sum3, _mm256_loadu_ps((const float*)(bias2 + 3))), _mm256_setzero_ps());
				_mm256_stream_ps(dptr + 24, sum3);
			}
		});
	}
	*/

#ifdef LEGACY
	template <int N>
	inline static __m128 vsum(const float4 *sptr, const float4 *fptr)
	{
		__m128 res = _mm_mul_ps(_mm_load_ps((const float*)(sptr++)), _mm_load_ps((const float*)(fptr++)));

		for (int i = 1; i < N; i++)
			res = _mm_add_ps(res, _mm_mul_ps(_mm_load_ps((const float*)(sptr++)), _mm_load_ps((const float*)(fptr++))));

		return res;
	}

	typedef __m128 __vfloat;
	inline static __m128 __vfloat_setzero_ps() { return _mm_setzero_ps(); }
	inline static __m128 __vfloat_add_ps(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
	inline static __m128 __vfloat_mul_ps(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
	inline static __m128 __vfloat_max_ps(__m128 a, __m128 b) { return _mm_max_ps(a, b); }
	inline static float __vfloat_sum(__m128 a) { return a.m128_f32[0] + a.m128_f32[1] + a.m128_f32[2] + a.m128_f32[3]; }
	inline static __m128 __vfloat_broadcast_ss(const void *mem) { return _mm_broadcast_ss((const float*)mem); }
	inline static __m128 __vfloat_load_ps(const void *mem) { return _mm_load_ps((const float*)mem); }
	inline static void __vfloat_stream_ps(void *mem, __m128 r) { _mm_stream_ps((float*)mem, r); }
#else
	template <int N>
	inline static __m256 vsum(const float8 *sptr, const float8 *fptr)
	{
		__m256 res = _mm256_mul_ps(_mm256_load_ps((const float*)(sptr++)), _mm256_load_ps((const float*)(fptr++)));

		for (int i = 1; i < N; i++)
			res = _mm256_add_ps(res, _mm256_mul_ps(_mm256_load_ps((const float*)(sptr++)), _mm256_load_ps((const float*)(fptr++))));

		return res;
	}

	typedef __m256 __vfloat;
	inline static __m256 __vfloat_setzero_ps() { return _mm256_setzero_ps(); }
	inline static __m256 __vfloat_add_ps(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
	inline static __m256 __vfloat_mul_ps(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
	inline static __m256 __vfloat_max_ps(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }
	inline static float __vfloat_sum(__m256 a) { return a.m256_f32[0] + a.m256_f32[1] + a.m256_f32[2] + a.m256_f32[3] + a.m256_f32[4] + a.m256_f32[5] + a.m256_f32[6] + a.m256_f32[7]; }
	inline static __m256 __vfloat_broadcast_ss(const void *mem) { return _mm256_broadcast_ss((const float*)mem); }
	inline static __m256 __vfloat_load_ps(const void *mem) { return _mm256_load_ps((const float*)mem); }
	inline static void __vfloat_stream_ps(void *mem, __m256 r) { _mm256_stream_ps((float*)mem, r); }
#endif

	void SRCNN_Resampling::ProcessLayer12(const Image<float> &src, Image3D<VectorFloat> &dst)
	{
		Image<VectorFloat> filter1(81, 64 / VectorFloat::size);
		Image<VectorFloat> filter2(32 / VectorFloat::size, 64);
		Image<VectorFloat> filter2a(64, 32 / VectorFloat::size);
		Image3D<VectorFloat> filter2b(4, 64, 8 / VectorFloat::size);
		VectorFloat bias1[64 / VectorFloat::size];
		VectorFloat bias2[32 / VectorFloat::size];

		for (int j = 0; j < 9; j++)
			for (int i = 0; i < 9; i++)
				for (int z = 0; z < 64; z++)
					filter1(j * 9 + i, z / VectorFloat::size).set(z % VectorFloat::size, weights_conv1[z][i][j]);

		for (int z = 0; z < 64; z++)
			bias1[z / VectorFloat::size].set(z % VectorFloat::size, biases1[z]);

		for (int z = 0; z < 64; z++)
			for (int k = 0; k < 32; k++)
			{
				int p = k / VectorFloat::size;
				filter2(p, z).set(k % VectorFloat::size, weights_conv2[k][0][0][z]);
				filter2a(z, p).set(k % VectorFloat::size, weights_conv2[k][0][0][z]);
				filter2b(p % 4, z, p / 4).set(k % VectorFloat::size, weights_conv2[k][0][0][z]);
			}

		for (int k = 0; k < 32; k++)
			bias2[k / VectorFloat::size].set(k % VectorFloat::size, biases2[k]);

		Parallel::For(0, src.Height(), [&src, &dst, &filter1, &bias1, &filter2b, &bias2](int y)
		{
			static const VectorFloat M1_255(1.0f / 255.0f);

			for (int x = 0; x < src.Width(); x++)
			{
				// ----- STEP 1 -----

				float v[81];

				if (x >= 4 && x < src.Width() - 4 && y >= 4 && y < src.Height() - 4)
				{
					for (int j = 0; j < 9; j++)
						for (int i = 0; i < 9; i++)
							v[j * 9 + i] = src(x + i - 4, y + j - 4);
				}
				else
				{
					for (int j = 0; j < 9; j++)
						for (int i = 0; i < 9; i++)
							v[j * 9 + i] = src(std::max(std::min(x + i - 4, src.Width() - 1), 0), std::max(std::min(y + j - 4, src.Height() - 1), 0));
				}

				VectorFloat tmp1[64 / VectorFloat::size];

				for (int z = 0; z < 64 / VectorFloat::size; z++)
				{
					VectorFloat s = VectorFloat::zero();

					const float *sptr = v;
					const VectorFloat *fptr = filter1.pixeladdr(0, z);

					for (int i = 0; i < 81; i++)
					{
						s = s + VectorFloat::broadcast(sptr++) * VectorFloat::load(fptr++);
					}

					VectorFloat m_bias = VectorFloat::load(bias1 + z);
					VectorFloat res = (s * M1_255 + m_bias).op_max(VectorFloat::zero());
					res.store(tmp1 + z);
				}

				// ----- STEP 2 -----

				VectorFloat *dptr = dst.pixeladdr(0, x, y);
				VectorFloat *pbias = bias2;

				for (int k = 0; k < 8 / VectorFloat::size; k++)
				{
					__vfloat sum0 = __vfloat_setzero_ps();
					__vfloat sum1 = __vfloat_setzero_ps();
					__vfloat sum2 = __vfloat_setzero_ps();
					__vfloat sum3 = __vfloat_setzero_ps();

					const float *sptr = (const float*)tmp1;
					const VectorFloat *fptr = filter2b.pixeladdr(0, 0, k);

					for (int z = 0; z < 64; z++)
					{
						__vfloat v = __vfloat_broadcast_ss(sptr++);
						sum0 = __vfloat_add_ps(sum0, __vfloat_mul_ps(v, __vfloat_load_ps(fptr++)));
						sum1 = __vfloat_add_ps(sum1, __vfloat_mul_ps(v, __vfloat_load_ps(fptr++)));
						sum2 = __vfloat_add_ps(sum2, __vfloat_mul_ps(v, __vfloat_load_ps(fptr++)));
						sum3 = __vfloat_add_ps(sum3, __vfloat_mul_ps(v, __vfloat_load_ps(fptr++)));
					}

					__vfloat r0 = __vfloat_max_ps(__vfloat_add_ps(sum0, __vfloat_load_ps(pbias++)), __vfloat_setzero_ps());
					__vfloat r1 = __vfloat_max_ps(__vfloat_add_ps(sum1, __vfloat_load_ps(pbias++)), __vfloat_setzero_ps());
					__vfloat r2 = __vfloat_max_ps(__vfloat_add_ps(sum2, __vfloat_load_ps(pbias++)), __vfloat_setzero_ps());
					__vfloat r3 = __vfloat_max_ps(__vfloat_add_ps(sum3, __vfloat_load_ps(pbias++)), __vfloat_setzero_ps());

					__vfloat_stream_ps(dptr++, r0);
					__vfloat_stream_ps(dptr++, r1);
					__vfloat_stream_ps(dptr++, r2);
					__vfloat_stream_ps(dptr++, r3);
				}
			}
		});
	}

	void SRCNN_Resampling::ProcessLayer3(const Image3D<VectorFloat> &src, Image<float> &dst)
	{
		Image<VectorFloat> filter(32, 25);

		for (int z = 0; z < 32; z++)
			for (int j = 0; j < 5; j++)
				for (int i = 0; i < 5; i++)
					filter(z / VectorFloat::size, j * 5 + i).set(z % VectorFloat::size, weights_conv3[i][j][z]);

		Parallel::For(0, dst.Height(), [this, &src, &dst, &filter](int y)
		{
			for (int x = 0; x < dst.Width(); x++)
			{
				__vfloat sum = __vfloat_setzero_ps();

				if (x >= 2 && x < dst.Width() - 2 && y >= 2 && y < dst.Height() - 2)
				{
					for (int j = 0; j < 5; j++)
						for (int i = 0; i < 5; i++)
						{
							const VectorFloat *sptr = (const VectorFloat*)src.pixeladdr(0, x + i - 2, y + j - 2);
							const VectorFloat *fptr = (const VectorFloat*)filter.pixeladdr(0, j * 5 + i);
							sum = __vfloat_add_ps(sum, vsum<32 / VectorFloat::size>(sptr, fptr));
						}
				}
				else
				{
					for (int j = 0; j < 5; j++)
						for (int i = 0; i < 5; i++)
						{
							const VectorFloat *sptr = (const VectorFloat*)src.pixeladdr(0, std::max(std::min(x + i - 2, dst.Width() - 1), 0), std::max(std::min(y + j - 2, dst.Height() - 1), 0));
							const VectorFloat *fptr = (const VectorFloat*)filter.pixeladdr(0, j * 5 + i);
							sum = __vfloat_add_ps(sum, vsum<32 / VectorFloat::size>(sptr, fptr));
						}
				}

				float res = __vfloat_sum(sum);
				dst(x, y) = f2b((res + biases3) * 255.0f);
			}
		});
	}

	void SRCNN_Resampling::Process(const ip::Image<float> &src, ip::Image<float> &dst, bool half_shift)
	{
		ip::Image<float> tmp(src.Width() * 2, src.Height() * 2);
		// ip::Image3D<float8> layer1(tmp.Width(), tmp.Height(), 8);
		ip::Image3D<VectorFloat> layer2(32 / VectorFloat::size, tmp.Width(), tmp.Height());

		BicubicInitialization(src, tmp);

		if (half_shift)
			BicubicInitialization2(src, tmp);

		auto t0 = std::chrono::high_resolution_clock::now();
		ProcessLayer12(tmp, layer2);
		auto t1 = std::chrono::high_resolution_clock::now();
		ProcessLayer3(layer2, dst);
		auto t2 = std::chrono::high_resolution_clock::now();

		printf("Step1 = %lld\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
		printf("Step2 = %lld\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
	}

	// =================================================================================================

	bool SRCNN_Resampling_x2(const ip::Image<float> &src, ip::Image<float> &dst, bool half_shift)
	{
		check(dst.Width() == src.Width() * 2 && dst.Height() == src.Height() * 2);

		std::unique_ptr<SRCNN_Resampling> res = make_unique<SRCNN_Resampling>();

		if (!res->LoadData(L"srcnn.bin"))
			return false;

		res->Process(src, dst, half_shift);

		// ip::Image<float> tmp(src.Width(), src.Height());
		// res->ProcessDebug(src, tmp);

		return true;
	}

	SRCNN::SRCNN(wchar_t *filename)
		: impl(new SRCNN_Resampling())
	{
		bool success = filename != nullptr ? impl->LoadData(filename) : impl->LoadData(L"srcnn.bin");
		if (!success)
		{
			delete impl;
			impl = nullptr;
		}
	}
		
	SRCNN::operator bool() const
	{
		return (bool)impl;
	}

	bool SRCNN::Resample_x2_915(const ip::Image<float> &src, ip::Image<float> &dst, bool half_shift)
	{
		if (!impl)
			return false;

		check(dst.Width() == src.Width() * 2 && dst.Height() == src.Height() * 2);
		impl->Process(src, dst, half_shift);

		return true;
	}

	SRCNN::~SRCNN()
	{
		delete impl;
	}
}
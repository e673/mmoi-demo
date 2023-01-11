#include "../../iplib/image/deblur/deblurtv.h"
#include "../../iplib/image/variational/varmethods.h"
#include <iplib/parallel.h>
#include <algorithm>
#include <immintrin.h>
#include <sstream>

namespace ip
{
	static Image<float> Expand(const Image<float> &src, int rx, int ry)
	{
		Image<float> res(src.Width() + 2 * rx, src.Height() + 2 * ry);

		for (int j = 0; j < res.Height(); j++)
			for (int i = 0; i < res.Width(); i++)
			{
				int x = (std::max)((std::min)(i - rx, src.Width() - 1), 0);
				int y = (std::max)((std::min)(j - ry, src.Height() - 1), 0);
				res(i, j) = src(x, y);
			}

		return res;
	}

	static void SetZero(Image<float> &img)
	{
		Parallel::For(0, img.Height(), [&img](int y)
		{
			for (int x = 0; x < img.Width(); x++)
				img(x, y) = 0.0f;
		});
	}

	static void SetZero(Image3D<float> &img)
	{
		Parallel::For(0, img.SizeZ(), [&img](int z)
		{
			for (int y = 0; y < img.SizeY(); y++)
				for (int x = 0; x < img.SizeX(); x++)
					img(x, y, z) = 0.0f;
		});
	}

	static void FillBorders(Image<float> &img, int rx, int ry)
	{
		for (int y = 0; y < ry; y++)
		{
			for (int x = 0; x < rx; x++)
				img(x, y) = img(rx, ry);

			for (int x = rx; x < img.Width() - rx; x++)
				img(x, y) = img(x, ry);

			for (int x = img.Width() - rx; x < img.Width(); x++)
				img(x, y) = img(img.Width() - rx - 1, ry);
		}

		Parallel::For(ry, img.Height() - ry, [&img, rx, ry](int y)
		{
			for (int x = 0; x < rx; x++)
				img(x, y) = img(rx, y);

			for (int x = img.Width() - rx; x < img.Width(); x++)
				img(x, y) = img(img.Width() - rx - 1, y);
		});

		for (int y = img.Height() - ry; y < img.Height(); y++)
		{
			for (int x = 0; x < rx; x++)
				img(x, y) = img(rx, img.Height() - ry - 1);

			for (int x = rx; x < img.Width() - rx; x++)
				img(x, y) = img(x, img.Height() - ry - 1);

			for (int x = img.Width() - rx; x < img.Width(); x++)
				img(x, y) = img(img.Width() - rx - 1, img.Height() - ry - 1);
		}
	}

	static void Convolve1(const Image<float> &img, const Image<float> &kernel, Image<float> &dst)
	{
		int rx = (kernel.Width() - 1) / 2;
		int ry = (kernel.Width() - 1) / 2;

		Parallel::For(0, dst.Height() - kernel.Height(), [&img, &kernel, &dst, rx, ry](int y)
		{
			for (int x = 0; x < dst.Width() - kernel.Width(); x++)
			{
				float s = 0.0f;

				for (int j = 0; j < kernel.Height(); j++)
					for (int i = 0; i < kernel.Width(); i++)
						s += img(x + i, y + j) * kernel(i, j);

				dst(x + rx, y + ry) = s;
			}
		});

		FillBorders(dst, rx, ry);
	}

	static void ConvolveLegacy(const Image<float> &img, const Image<float> &kernel, Image<float> &dst)
	{
		int rx = (kernel.Width() - 1) / 2;
		int ry = (kernel.Width() - 1) / 2;

		Parallel::For(0, dst.Height(), [&img, &kernel, &dst, rx, ry](int y)
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
		});
	}

	// =======================================================================================================

	class ConvolveClass3D
	{
	public:
		ConvolveClass3D(const Image3D<float> &img, const Image3D<float> &kernel, Image3D<float> &dst);
		void Perform();

	private:
		const Image3D<float> &img;
		const Image3D<float> &kernel;
		Image3D<float> &dst;
		int rx, ry, rz;

	private:
		void ProcessCheckedPlane(int z);
		void ProcessUncheckedPlane(int z);
		inline void ProcessUncheckedLineBorder(int x, int y, int z);
	};

	ConvolveClass3D::ConvolveClass3D(const Image3D<float> &img, const Image3D<float> &kernel, Image3D<float> &dst)
		: img(img), kernel(kernel), dst(dst), rx(kernel.SizeX() / 2), ry(kernel.SizeY() / 2), rz(kernel.SizeZ() / 2) {}

	void ConvolveClass3D::ProcessCheckedPlane(int z)
	{
		for (int y = 0; y < dst.SizeY(); y++)
		{
			for (int x = 0; x < dst.SizeX(); x++)
			{
				float s = 0.0f;

				for (int k = 0; k < kernel.SizeZ(); k++)
					for (int j = 0; j < kernel.SizeY(); j++)
						for (int i = 0; i < kernel.SizeX(); i++)
							s += img((std::min)((std::max)(x + i - rx, 0), dst.SizeX() - 1),
							         (std::min)((std::max)(y + j - ry, 0), dst.SizeY() - 1),
								     (std::min)((std::max)(z + k - rz, 0), dst.SizeZ() - 1)) * kernel(i, j, k);

				dst(x, y, z) = s;
			}
		}
	}

	inline void ConvolveClass3D::ProcessUncheckedLineBorder(int x, int y, int z)
	{
		float s = 0.0f;

		for (int k = 0; k < kernel.SizeZ(); k++)
			for (int j = 0; j < kernel.SizeY(); j++)
				for (int i = 0; i < kernel.SizeX(); i++)
					s += img((std::min)((std::max)(x + i - rx, 0), dst.SizeX() - 1),
					         (std::min)((std::max)(y + j - ry, 0), dst.SizeY() - 1),
							 z + k - rz) * kernel(i, j, k);

		dst(x, y, z) = s;
	}

	void ConvolveClass3D::ProcessUncheckedPlane(int z)
	{
		for (int y = 0; y < ry; y++)
			for (int x = 0; x < dst.SizeX(); x++)
				ProcessUncheckedLineBorder(x, y, z);

		for (int y = ry; y < dst.SizeY() - ry; y++)
		{
			int x = 0;

			while (x < rx)
				ProcessUncheckedLineBorder(x++, y, z);

			while (x < dst.SizeX() - rx - 7)
			{
				__m256 s = _mm256_setzero_ps();

				for (int k = 0; k < kernel.SizeZ(); k++)
					for (int j = 0; j < kernel.SizeY(); j++)
						for (int i = 0; i < kernel.SizeX(); i++)
						{
							__m256 p = _mm256_load_ps(img.pixeladdr(x + i - rx, y + j - ry, z + k - rz));
							__m256 q = _mm256_broadcast_ss(kernel.pixeladdr(i, j, k));
							s = _mm256_add_ps(s, _mm256_mul_ps(p, q));
						}

				_mm256_store_ps(dst.pixeladdr(x, y, z), s);

				x += 8;
			}

			while (x < dst.SizeX())
				ProcessUncheckedLineBorder(x++, y, z);
		}

		for (int y = dst.SizeY() - ry; y < dst.SizeY(); y++)
			for (int x = 0; x < dst.SizeX(); x++)
				ProcessUncheckedLineBorder(x, y, z);
	}

	void ConvolveClass3D::Perform()
	{
		_mm256_zeroall();

		Parallel::For(0, dst.SizeZ(), [this](int z)
		{
			if (z < rz || z >= dst.SizeZ() - rz)
				ProcessCheckedPlane(z);
			else
				ProcessUncheckedPlane(z);
		});

		_mm256_zeroall();
	}

	static void Convolve(const Image3D<float> &img, const Image3D<float> &kernel, Image3D<float> &dst)
	{
		ConvolveClass3D(img, kernel, dst).Perform();
	}

	// =======================================================================================================

	static void Discrepancy(const Image3D<float> &v, const Image3D<float> &u, const Image3D<float> &kernel, Image3D<float> &tmp, Image3D<float> &dst)
	{
		Convolve(v, kernel, tmp);

		Parallel::For(0, tmp.SizeZ(), [&tmp, &u](int z)
		{
			for (int y = 0; y < tmp.SizeY(); y++)
				for (int x = 0; x < tmp.SizeX(); x++)
					tmp(x, y, z) -= u(x, y, z);
		});

		Convolve(tmp, kernel, dst);
	}

	inline static float sign(float x)
	{
		if (x > 0.0f)
			return 1.0f;
		else if (x < 0.0f)
			return -1.0f;
		else
			return 0.0f;
	}

	#pragma region TVGradient

	static inline float TVGradientCenter(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x - 1, y)) + sign(v - src(x + 1, y)) + sign(v - src(x, y - 1)) + sign(v - src(x, y + 1));
	}

	static inline float TVGradientLeft(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x + 1, y)) + sign(v - src(x, y - 1)) + sign(v - src(x, y + 1));
	}

	static inline float TVGradientRight(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x - 1, y)) + sign(v - src(x, y - 1)) + sign(v - src(x, y + 1));
	}

	static inline float TVGradientTop(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x - 1, y)) + sign(v - src(x + 1, y)) + sign(v - src(x, y + 1));
	}

	static inline float TVGradientBottom(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x - 1, y)) + sign(v - src(x + 1, y)) + sign(v - src(x, y - 1));
	}

	static inline float TVGradientTopLeft(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x + 1, y)) + sign(v - src(x, y + 1));
	}

	static inline float TVGradientTopRight(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x - 1, y)) + sign(v - src(x, y + 1));
	}

	static inline float TVGradientBottomLeft(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x + 1, y)) + sign(v - src(x, y - 1));
	}

	static inline float TVGradientBottomRight(const Image<float> &src, int x, int y)
	{
		float v = src(x, y);
		return sign(v - src(x - 1, y)) + sign(v - src(x, y - 1));
	}

	static void TVGradient(const Image<float> &src, Image<float> &dst, float alpha)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());

		dst(0, 0) += TVGradientTopLeft(src, 0, 0) * alpha;
		
		for (int i = 1; i < src.Width() - 1; i++)
			dst(i, 0) += TVGradientTop(src, i, 0) * alpha;

		dst(src.Width() - 1, 0) += TVGradientTopRight(src, src.Width() - 1, 0) * alpha;

		for (int j = 1; j < src.Height() - 1; j++)
		{
			dst(0, j) += TVGradientLeft(src, 0, j) * alpha;

			for (int i = 1; i < src.Width() - 1; i++)
				dst(i, j) += TVGradientCenter(src, i, j) * alpha;

			dst(src.Width() - 1, j) += TVGradientRight(src, src.Width() - 1, j) * alpha;
		}

		dst(0, src.Height() - 1) += TVGradientBottomLeft(src, 0, src.Height() - 1) * alpha;

		for (int i = 1; i < src.Width() - 1; i++)
			dst(i, src.Height() - 1) += TVGradientBottom(src, i, src.Height() - 1) * alpha;

		dst(src.Width() - 1, src.Height() - 1) += TVGradientBottomRight(src, src.Width() - 1, src.Height() - 1) * alpha;

	}

	#pragma endregion

	static inline float BTVGradient3DSafe(const Image3D<float> &src, int x, int y, int z)
	{
		float v = src(x, y, z);

		float p = 0.0f;

		for (int k = -1; k <= 1; k++)
			for (int j = -1; j <= 1; j++)
				for (int i = -1; i <= 1; i++)
				{
					if (x + i < 0 || x + i >= src.SizeX() || y + j < 0 || y + j >= src.SizeY() || z + k < 0 || z + k >= src.SizeZ())
						continue;

					float c = src(x + i, y + j, z + k);

					float q = 1.0f;

					if (i != 0)
						q *= 0.8f;

					if (j != 0)
						q *= 0.8f;

					if (k != 0)
						q *= 0.8f;

					p += sign(v - c) * q;
				}

		return p;
	}

	static inline float BTVGradient3DUnsafe(const Image3D<float> &src, int x, int y, int z)
	{
		float v = src(x, y, z);

		float p1 =
			sign(v - src(x - 1, y, z)) + sign(v - src(x + 1, y, z)) +
			sign(v - src(x, y - 1, z)) + sign(v - src(x, y + 1, z)) +
			sign(v - src(x, y, z - 1)) + sign(v - src(x, y, z + 1));

		float p2 =
			sign(v - src(x, y - 1, z - 1)) + sign(v - src(x, y + 1, z - 1)) + sign(v - src(x, y - 1, z + 1)) + sign(v - src(x, y + 1, z + 1)) +
			sign(v - src(x - 1, y, z - 1)) + sign(v - src(x + 1, y, z - 1)) + sign(v - src(x - 1, y, z + 1)) + sign(v - src(x + 1, y, z + 1)) +
			sign(v - src(x - 1, y - 1, z)) + sign(v - src(x + 1, y - 1, z)) + sign(v - src(x - 1, y + 1, z)) + sign(v - src(x + 1, y + 1, z));

		float p3 =
			sign(v - src(x - 1, y - 1, z - 1)) + sign(v - src(x + 1, y - 1, z - 1)) + sign(v - src(x - 1, y + 1, z - 1)) + sign(v - src(x + 1, y + 1, z - 1)) +
			sign(v - src(x - 1, y - 1, z + 1)) + sign(v - src(x + 1, y - 1, z + 1)) + sign(v - src(x - 1, y + 1, z + 1)) + sign(v - src(x + 1, y + 1, z + 1));

		return p1 * 0.8f + p2 * 0.64f + p3 * 0.512f;
	}

	static void BTVGradient3D(const Image3D<float> &src, Image3D<float> &dst, float alpha)
	{
		check(src.SizeX() == dst.SizeX() && src.SizeY() == dst.SizeY() && src.SizeZ() == dst.SizeZ());

		for (int j = 0; j < src.SizeY(); j++)
			for (int i = 0; i < src.SizeX(); i++)
				dst(i, j, 0) += BTVGradient3DSafe(src, i, j, 0) * alpha;

		Parallel::For(1, src.SizeZ() - 1, [&src, &dst, alpha](int k)
		{
			for (int i = 0; i < src.SizeX(); i++)
				dst(i, 0, k) += BTVGradient3DSafe(src, i, 0, k) * alpha;

			for (int j = 1; j < src.SizeY() - 1; j++)
			{
				dst(0, j, k) += BTVGradient3DSafe(src, 0, j, k) * alpha;

				for (int i = 1; i < src.SizeX() - 1; i++)
					dst(i, j, k) += BTVGradient3DUnsafe(src, i, j, k) * alpha;

				dst(src.SizeX() - 1, j, k) += BTVGradient3DSafe(src, src.SizeX() - 1, j, k) * alpha;
			}

			for (int i = 0; i < src.SizeX(); i++)
				dst(i, src.SizeY() - 1, k) += BTVGradient3DSafe(src, i, src.SizeY() - 1, k) * alpha;
		});

		for (int j = 0; j < src.SizeY(); j++)
			for (int i = 0; i < src.SizeX(); i++)
				dst(i, j, src.SizeZ() - 1) += BTVGradient3DSafe(src, i, j, src.SizeZ() - 1) * alpha;
	}

	/* static inline float BTVGradientSafe2(const Image<float> &src, int x, int y, int dx, int dy)
	{
		float p1 = src((std::min)((std::max)(x - 2 * dx, 0), src.Width() - 1), (std::min)((std::max)(y - 2 * dy, 0), src.Height() - 1));
		float p2 = src((std::min)((std::max)(x - dx, 0), src.Width() - 1), (std::min)((std::max)(y - dy, 0), src.Height() - 1));
		float p3 = src(x, y);
		float p4 = src((std::min)((std::max)(x + dx, 0), src.Width() - 1), (std::min)((std::max)(y + dy, 0), src.Height() - 1));
		float p5 = src((std::min)((std::max)(x + 2 * dx, 0), src.Width() - 1), (std::min)((std::max)(y + 2 * dy, 0), src.Height() - 1));

		return 2.0f * sign(2.0f * p3 - p2 - p4) + sign(p1 + p3 - 2.0f * p2) + sign(p5 + p3 - 2.0f * p4);
	} */

	static inline float BTVGradient3DSafe2(const Image3D<float> &src, int x, int y, int z, int dx, int dy, int dz)
	{
		float p1 = src(
			(std::min)((std::max)(x - 2 * dx, 0), src.SizeX() - 1),
			(std::min)((std::max)(y - 2 * dy, 0), src.SizeY() - 1),
			(std::min)((std::max)(z - 2 * dz, 0), src.SizeZ() - 1));

		float p2 = src(
			(std::min)((std::max)(x - dx, 0), src.SizeX() - 1),
			(std::min)((std::max)(y - dy, 0), src.SizeY() - 1),
			(std::min)((std::max)(z - dz, 0), src.SizeZ() - 1));

		float p3 = src(x, y, z);

		float p4 = src(
			(std::min)((std::max)(x + dx, 0), src.SizeX() - 1),
			(std::min)((std::max)(y + dy, 0), src.SizeY() - 1),
			(std::min)((std::max)(z + dz, 0), src.SizeZ() - 1));

		float p5 = src(
			(std::min)((std::max)(x + 2 * dx, 0), src.SizeX() - 1),
			(std::min)((std::max)(y + 2 * dy, 0), src.SizeY() - 1),
			(std::min)((std::max)(z + 2 * dz, 0), src.SizeZ() - 1));

		return 2.0f * sign(2.0f * p3 - p2 - p4) + sign(p1 + p3 - 2.0f * p2) + sign(p5 + p3 - 2.0f * p4);
	}

	/* static inline float BTVGradientUnsafe2(const Image<float> &src, int x, int y, int dx, int dy)
	{
		float p1 = src(x - 2 * dx, y - 2 * dy);
		float p2 = src(x - dx, y - dy);
		float p3 = src(x, y);
		float p4 = src(x + dx, y + dy);
		float p5 = src(x + 2 * dx, y + 2 * dy);

		return 2.0f * sign(2.0f * p3 - p2 - p4) + sign(p1 + p3 - 2.0f * p2) + sign(p5 + p3 - 2.0f * p4);
	} */

	static inline float BTVGradient3DUnsafe2(const Image3D<float> &src, int x, int y, int z, int dx, int dy, int dz)
	{
		float p1 = src(x - 2 * dx, y - 2 * dy, z - 2 * dz);
		float p2 = src(x - dx, y - dy, z - dz);
		float p3 = src(x, y, z);
		float p4 = src(x + dx, y + dy, z + dz);
		float p5 = src(x + 2 * dx, y + 2 * dy, z + 2 * dz);

		return 2.0f * sign(2.0f * p3 - p2 - p4) + sign(p1 + p3 - 2.0f * p2) + sign(p5 + p3 - 2.0f * p4);
	}

	/* static inline float BTVGradientSafe2(const Image<float> &src, int x, int y)
	{
		float v1 = BTVGradientSafe2(src, x, y, 1, 0);
		float v2 = BTVGradientSafe2(src, x, y, 0, 1);
		float v3 = BTVGradientSafe2(src, x, y, 1, 1);
		float v4 = BTVGradientSafe2(src, x, y, -1, 1);

		return v1 + v2 + (v3 + v4) * 0.5f;
	} */

	static inline float BTVGradient3DSafe2(const Image3D<float> &src, int x, int y, int z)
	{
		float a1 = BTVGradient3DSafe2(src, x, y, z, 1, 0, 0);
		float a2 = BTVGradient3DSafe2(src, x, y, z, 0, 1, 0);
		float a3 = BTVGradient3DSafe2(src, x, y, z, 0, 0, 1);

		float b1 = BTVGradient3DSafe2(src, x, y, z, 1, -1, 0);
		float b2 = BTVGradient3DSafe2(src, x, y, z, 1, 1, 0);
		float b3 = BTVGradient3DSafe2(src, x, y, z, 1, 0, -1);
		float b4 = BTVGradient3DSafe2(src, x, y, z, 1, 0, 1);
		float b5 = BTVGradient3DSafe2(src, x, y, z, 0, 1, -1);
		float b6 = BTVGradient3DSafe2(src, x, y, z, 0, 1, 1);

		float c1 = BTVGradient3DSafe2(src, x, y, z, 1, 1, 1);
		float c2 = BTVGradient3DSafe2(src, x, y, z, 1, 1, -1);
		float c3 = BTVGradient3DSafe2(src, x, y, z, 1, -1, 1);
		float c4 = BTVGradient3DSafe2(src, x, y, z, 1, -1, -1);

		return a1 + a2 + a3 + (b1 + b2 + b3 + b4 + b5 + b6) * 0.5f + (c1 + c2 + c3 + c4) * 0.25f;
	}

	/* static inline float BTVGradientUnsafe2(const Image<float> &src, int x, int y)
	{
		float v1 = BTVGradientUnsafe2(src, x, y, 1, 0);
		float v2 = BTVGradientUnsafe2(src, x, y, 0, 1);
		float v3 = BTVGradientUnsafe2(src, x, y, 1, 1);
		float v4 = BTVGradientUnsafe2(src, x, y, -1, 1);

		return v1 + v2 + (v3 + v4) * 0.5f;
	} */

	static inline float BTVGradient3DUnsafe2(const Image3D<float> &src, int x, int y, int z)
	{
		float a1 = BTVGradient3DUnsafe2(src, x, y, z, 1, 0, 0);
		float a2 = BTVGradient3DUnsafe2(src, x, y, z, 0, 1, 0);
		float a3 = BTVGradient3DUnsafe2(src, x, y, z, 0, 0, 1);

		float b1 = BTVGradient3DUnsafe2(src, x, y, z, 1, -1, 0);
		float b2 = BTVGradient3DUnsafe2(src, x, y, z, 1, 1, 0);
		float b3 = BTVGradient3DUnsafe2(src, x, y, z, 1, 0, -1);
		float b4 = BTVGradient3DUnsafe2(src, x, y, z, 1, 0, 1);
		float b5 = BTVGradient3DUnsafe2(src, x, y, z, 0, 1, -1);
		float b6 = BTVGradient3DUnsafe2(src, x, y, z, 0, 1, 1);

		float c1 = BTVGradient3DUnsafe2(src, x, y, z, 1, 1, 1);
		float c2 = BTVGradient3DUnsafe2(src, x, y, z, 1, 1, -1);
		float c3 = BTVGradient3DUnsafe2(src, x, y, z, 1, -1, 1);
		float c4 = BTVGradient3DUnsafe2(src, x, y, z, 1, -1, -1);

		return a1 + a2 + a3 + (b1 + b2 + b3 + b4 + b5 + b6) * 0.5f + (c1 + c2 + c3 + c4) * 0.25f;
	}

	/* static void BTVGradient2(const Image<float> &src, Image<float> &dst, float alpha)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());

		for (int j = 0; j < 2; j++)
			for (int i = 0; i < src.Width(); i++)
				dst(i, j) += BTVGradientSafe2(src, i, j) * alpha;

		for (int j = 2; j < src.Height() - 2; j++)
		{
			dst(0, j) += BTVGradientSafe2(src, 0, j) * alpha;
			dst(1, j) += BTVGradientSafe2(src, 1, j) * alpha;

			for (int i = 2; i < src.Width() - 2; i++)
				dst(i, j) += BTVGradientUnsafe2(src, i, j) * alpha;

			dst(src.Width() - 2, j) += BTVGradientSafe2(src, src.Width() - 2, j) * alpha;
			dst(src.Width() - 1, j) += BTVGradientSafe2(src, src.Width() - 1, j) * alpha;
		}

		for (int j = src.Height() - 2; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				dst(i, j) += BTVGradientSafe2(src, i, j) * alpha;
	} */

	static void BTVGradient3D2(const Image3D<float> &src, Image3D<float> &dst, float alpha)
	{
		check(src.SizeX() == dst.SizeX() && src.SizeY() == dst.SizeY() && src.SizeZ() == dst.SizeZ());

		for (int k = 0; k < 2; k++)
			for (int j = 0; j < src.SizeY(); j++)
				for (int i = 0; i < src.SizeX(); i++)
					dst(i, j, k) += BTVGradient3DSafe2(src, i, j, k) * alpha;

		Parallel::For(2, src.SizeZ() - 2, [&src, &dst, alpha] (int k)
		{
			for (int j = 0; j < 2; j++)
				for (int i = 0; i < src.SizeX(); i++)
					dst(i, j, k) += BTVGradient3DSafe2(src, i, j, k) * alpha;

			for (int j = 2; j < src.SizeY() - 2; j++)
			{
				dst(0, j, k) += BTVGradient3DSafe2(src, 0, j, k) * alpha;
				dst(1, j, k) += BTVGradient3DSafe2(src, 1, j, k) * alpha;

				for (int i = 2; i < src.SizeX() - 2; i++)
					dst(i, j, k) += BTVGradient3DUnsafe2(src, i, j, k) * alpha;

				dst(src.SizeX() - 2, j, k) += BTVGradient3DSafe2(src, src.SizeX() - 2, j, k) * alpha;
				dst(src.SizeX() - 1, j, k) += BTVGradient3DSafe2(src, src.SizeX() - 1, j, k) * alpha;
			}

			for (int j = src.SizeY() - 2; j < src.SizeY(); j++)
				for (int i = 0; i < src.SizeX(); i++)
					dst(i, j, k) += BTVGradient3DSafe2(src, i, j, k) * alpha;
		});

		for (int k = src.SizeZ() - 2; k < src.SizeZ(); k++)
			for (int j = 0; j < src.SizeY(); j++)
				for (int i = 0; i < src.SizeX(); i++)
					dst(i, j, k) += BTVGradient3DSafe2(src, i, j, k) * alpha;
	}

	static float NormL1(const Image3D<float> &img)
	{
		float res = 0.0f;

		for (int k = 0; k < img.SizeZ(); k++)
			for (int j = 0; j < img.SizeY(); j++)
				for (int i = 0; i < img.SizeX(); i++)
					res += fabsf(img(i, j, k));

		return res / (img.SizeX() * img.SizeY() * img.SizeZ());
	}

	static void ApplyGradient(Image<float> &dst, const Image<float> &grad, float q)
	{
		check(grad.Width() == dst.Width() && grad.Height() == dst.Height());

		Parallel::For(0, grad.Height(), [&grad, &dst, q](int y)
		{
			for (int x = 0; x < grad.Width(); x++)
			{
				dst(x, y) -= grad(x, y) * q;
			}
		});
	}

	/* static void WriteResult(const Image<float> &src, Image<float> &dst)
	{
		int ofsx = (src.Width() - dst.Width()) / 2;
		int ofsy = (src.Height() - dst.Height()) / 2;

		Parallel::For(0, dst.Height(), [&src, &dst, ofsx, ofsy](int y)
		{
			for (int x = 0; x < dst.Width(); x++)
				dst(x, y) = src(x + ofsx, y + ofsy);
		});
	} */

	void DeblurTV::AnyKernel(const Image<float> &src, Image<float> &dst, const Image<float> &kernel, float reg_param)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());
		check(kernel.Width() % 2 == 1 && kernel.Height() % 2 == 1);

		int rx = (kernel.Width() - 1) / 2;
		int ry = (kernel.Height() - 1) / 2;

		Image<float> temp(src.Width(), src.Height()); // Single padding
		Image<float> gradient(src.Width(), src.Height());

		SetZero(dst);

		for (int iter = 0; iter < 100; iter++)
		{
			float step = powf(0.95f, (float)iter) * 25.0f;
			VarMethods::DiscrepancyL2(dst, src, kernel, gradient, temp);
			VarMethods::AddGradientBTV(dst, gradient, reg_param);
			float norm = VarMethods::CalcNormL1(gradient);
			ApplyGradient(dst, gradient, step / norm);

			// printf("Iter = %d, norm = %f\n", iter, norm);
		}
	}

	static float NormalizeGradient(Image3D<float> &grad, float target_norm = 1.0f)
	{
		float norm = NormL1(grad);
		float q = target_norm / norm;

		Parallel::For(0, grad.SizeZ(), [&grad, q](int z)
		{
			for (int y = 0; y < grad.SizeY(); y++)
				for (int x = 0; x < grad.SizeX(); x++)
					grad(x, y, z) *= q;
		});

		return norm;
	}

	static void ApplyMomentum(Image3D<float> &dst, const Image3D<float> &cur, Image3D<float> &prev, float mu)
	{
		Parallel::For(0, dst.SizeZ(), [&cur, &prev, &dst, mu](int z)
		{
			for (int y = 0; y < dst.SizeY(); y++)
				for (int x = 0; x < dst.SizeX(); x++)
				{
					float v = prev(x, y, z) * mu + cur(x, y, z);
					prev(x, y, z) = v;
					dst(x, y, z) -= v;
				}
		});
	}

	static void ApplyNormalizedMomentum(Image<float> &dst, const Image<float> &cur, Image<float> &prev, float mu, float step)
	{
		Parallel::For(0, dst.Height(), [&cur, &prev, &dst, mu](int y)
		{
			for (int x = 0; x < dst.Width(); x++)
			{
				prev(x, y) = prev(x, y) * mu + cur(x, y);
			}
		});

		VarMethods::NormalizeGradientL1(prev, step);

		Parallel::For(0, dst.Height(), [&cur, &prev, &dst, mu](int y)
		{
			for (int x = 0; x < dst.Width(); x++)
			{
				dst(x, y) -= prev(x, y);
			}
		});
	}

	static void Subtract(Image3D<float> &dst, const Image3D<float> &src1, const Image3D<float> &src2, float q)
	{
		Parallel::For(0, dst.SizeZ(), [&src1, &src2, &dst, q](int z)
		{
			for (int y = 0; y < dst.SizeY(); y++)
				for (int x = 0; x < dst.SizeX(); x++)
					dst(x, y, z) = src1(x, y, z) - src2(x, y, z) * q;
		});
	}

	void DeblurTV::AnyKernel2(const Image<float> &src, Image<float> &dst, const Image<float> &kernel, float reg_param_1, float reg_param_2, int num_iter, float mu, float corr_factor)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());
		check(kernel.Width() % 2 == 1 && kernel.Height() % 2 == 1);

		int rx = (kernel.Width() - 1) / 2;
		int ry = (kernel.Height() - 1) / 2;

		Image<float> temp(src.Width(), src.Height()); // Single padding
		Image<float> cur_gradient(src.Width(), src.Height());
		Image<float> prev_gradient(src.Width(), src.Height());
		Image<float> x(dst.Width(), dst.Height());

		SetZero(dst);
		SetZero(prev_gradient);

		for (int iter = 0; iter < num_iter; iter++)
		{
			float step = powf(0.01f * corr_factor, (float)iter / num_iter) * 25.0f;

			VarMethods::Subtract(x, dst, prev_gradient, mu);
			VarMethods::DiscrepancyL2(x, src, kernel, cur_gradient, temp);
			VarMethods::AddGradientBTV(x, cur_gradient, reg_param_1);
			VarMethods::AddGradientBTV2(x, cur_gradient, reg_param_2);
			VarMethods::NormalizeGradientL1(cur_gradient, step);
			VarMethods::ApplyMomentum(dst, cur_gradient, prev_gradient, mu);
		}
	}

	void DeblurTV::AnyKernel3(const Image3D<float> &src, Image3D<float> &dst, const Image3D<float> &kernel, float reg_param_1, float reg_param_2, int num_iter, float mu)
	{
		check(src.SizeX() == dst.SizeX() && src.SizeY() == dst.SizeY() && src.SizeZ() == dst.SizeZ());
		check(kernel.SizeX() % 2 == 1 && kernel.SizeY() % 2 == 1 && kernel.SizeZ() % 2 == 1);

		int rx = (kernel.SizeX() - 1) / 2;
		int ry = (kernel.SizeY() - 1) / 2;
		int rz = (kernel.SizeZ() - 1) / 2;

		Image3D<float> temp(src.SizeX(), src.SizeY(), src.SizeZ()); // Single padding
		Image3D<float> cur_gradient(src.SizeX(), src.SizeY(), src.SizeZ());
		Image3D<float> prev_gradient(src.SizeX(), src.SizeY(), src.SizeZ());
		Image3D<float> x(dst.SizeX(), dst.SizeY(), dst.SizeZ());

		SetZero(dst);
		SetZero(prev_gradient);

		for (int iter = 0; iter < num_iter; iter++)
		{
			float step = powf(0.01f, (float)iter / num_iter) * 25.0f;

			Subtract(x, dst, prev_gradient, mu);
			Discrepancy(x, src, kernel, temp, cur_gradient);
			BTVGradient3D(x, cur_gradient, reg_param_1);
			BTVGradient3D2(x, cur_gradient, reg_param_2);
			float norm = NormalizeGradient(cur_gradient, step);
			ApplyMomentum(dst, cur_gradient, prev_gradient, mu);

			printf("Iter = %d, norm = %f.3\n", iter, norm);
		}
	}
}
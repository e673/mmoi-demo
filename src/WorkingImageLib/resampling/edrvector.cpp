#include "edrvector.h"

#include <iplib/math/quadratic_optimization.h>
#include <misc/blocksplit.h>
#include <iplib/parallel.h>

 #include <Windows.h>

namespace ip
{
	class EDRVector::Impl
	{
	public:
		Image<VectorByte> Perform(const VectorImageFloatColor &padded_lr);
		void Perform(const VectorImageFloatColor &padded_lr, Image<VectorByte> &res);

		void LearnStep1(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr);
		void UpdateCoefficientsStep1();

		void LearnStep2(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr);
		void UpdateCoefficientsStep2();

	private:
		float kernel0[6] = { -0.00063f, -0.00180f, 0.02694f, 0.01716f, -0.15097f, 1.45306f };
		float kernel1[6] = { 0.03297f, -0.04665f, -0.04484f, -0.05382f, 0.12650f, 0.58139f };
		float kernel2[6] = { -0.03016f, -0.04952f, -0.05838f, 0.04283f, 0.45204f, 0.25095f };
		float WeightThreshold = 1.25f;

		QuadraticOptimization opt0, opt1, opt2;

		const static VectorFloat REDQ, GREENQ, BLUEQ, ONE;
		const static VectorFloat SIGNMASK;

	private:

		void ProcessStep0(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &dst, Image<VectorByte> &res);
		void ProcessStep1(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &dst, Image<VectorByte> &res);
		void ProcessStep2(VectorImageFloatColor &dst, Image<VectorByte> &res);

		inline static VectorFloat CalcWeights(VectorFloat p, VectorFloat q);

		inline static VectorFloat abs(VectorFloat value);
		inline static VectorFloat AverageNormal3x3(const VectorImageFloat &img, int x, int y);
		inline static VectorFloat AverageAbnormal3x3_dir1(const VectorImageFloatColor &img, int x, int y);
		inline static VectorFloat AverageDiag3x3(const VectorImageFloat &img, int x, int y);
	};

	// ==================================================================================================

	const VectorFloat EDRVector::Impl::REDQ(0.299f);
	const VectorFloat EDRVector::Impl::GREENQ(0.587f);
	const VectorFloat EDRVector::Impl::BLUEQ(0.114f);
	const VectorFloat EDRVector::Impl::ONE(1.0f);
#ifdef LEGACY
	const VectorFloat EDRVector::Impl::SIGNMASK(_mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
#else
	const VectorFloat EDRVector::Impl::SIGNMASK(_mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
#endif

	Image<VectorByte> EDRVector::Impl::Perform(const VectorImageFloatColor &padded_lr)
	{
		Image<VectorByte> res(padded_lr.Width() * 2, padded_lr.Height() * 2);
		Perform(padded_lr, res);
		return res;
	}

	void EDRVector::Impl::Perform(const VectorImageFloatColor &padded_lr, Image<VectorByte> &res)
	{
		check(padded_lr.Width() * 2 == res.Width() && padded_lr.Height() * 2 == res.Height());

		VectorImageFloatColor dst(padded_lr.Width() * 2, padded_lr.Height() * 2);

		ProcessStep0(padded_lr, dst, res);
		ProcessStep1(padded_lr, dst, res);
		ProcessStep2(dst, res);
	}

	void EDRVector::Impl::LearnStep1(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr)
	{
		die("Not implemented");
	}

	void EDRVector::Impl::UpdateCoefficientsStep1()
	{
		die("Not implemented");
	}

	void EDRVector::Impl::LearnStep2(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr)
	{
		die("Not implemented");
	}

	void EDRVector::Impl::UpdateCoefficientsStep2()
	{
		die("Not implemented");
	}

	void EDRVector::Impl::ProcessStep0(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &dst, Image<VectorByte> &res)
	{
		Parallel::For(2, padded_lr.Height() - 2, [&padded_lr, &dst, &res, this](int j)
		{
			for (int i = 2; i < padded_lr.Width() - 2; i++)
			{
				VectorFloatColor v0 = padded_lr(i - 2, j - 2) + padded_lr(i + 2, j - 2) + padded_lr(i - 2, j + 2) + padded_lr(i + 2, j + 2);
				VectorFloatColor v1 = padded_lr(i - 1, j - 2) + padded_lr(i + 1, j - 2) + padded_lr(i - 2, j - 1) + padded_lr(i + 2, j - 1) +
					padded_lr(i - 2, j + 1) + padded_lr(i + 2, j + 1) + padded_lr(i - 1, j + 2) + padded_lr(i + 1, j + 2);
				VectorFloatColor v2 = padded_lr(i, j - 2) + padded_lr(i - 2, j) + padded_lr(i + 2, j) + padded_lr(i, j + 2);
				VectorFloatColor v3 = padded_lr(i - 1, j - 1) + padded_lr(i + 1, j - 1) + padded_lr(i - 1, j + 1) + padded_lr(i + 1, j + 1);
				VectorFloatColor v4 = padded_lr(i, j - 1) + padded_lr(i - 1, j) + padded_lr(i + 1, j) + padded_lr(i, j + 1);
				VectorFloatColor v5 = padded_lr(i, j);

				VectorFloatColor c = v0 * kernel0[0] + v1 * kernel0[1] + v2 * kernel0[2] + v3 * kernel0[3] + v4 * kernel0[4] + v5 * kernel0[5];

				dst(i * 2, j * 2) = c;
				res(i * 2, j * 2) = c.ToByte();
			}
		});
	}

	VectorFloat EDRVector::Impl::AverageNormal3x3(const VectorImageFloat &img, int x, int y)
	{
		return img(x - 1, y - 1) + img(x, y - 1) + img(x + 1, y - 1) + img(x - 1, y) + img(x, y) + img(x + 1, y) + img(x - 1, y + 1) + img(x, y + 1) + img(x + 1, y + 1);
	}

	VectorFloat EDRVector::Impl::AverageDiag3x3(const VectorImageFloat &img, int x, int y)
	{
		return img(x, y - 2) + img(x - 1, y - 1) + img(x + 1, y - 1) + img(x - 2, y) + img(x, y) + img(x + 2, y) + img(x - 1, y + 1) + img(x + 1, y + 1) + img(x, y + 2);
	}

	void EDRVector::Impl::ProcessStep1(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &dst, Image<VectorByte> &res)
	{
		VectorImageFloat dir1(padded_lr.Width() - 1, padded_lr.Height() - 1);
		VectorImageFloat dir2(padded_lr.Width() - 1, padded_lr.Height() - 1);

		Parallel::For(1, padded_lr.Height(), [&padded_lr, &dir1, &dir2](int j)
		{
			for (int i = 1; i < padded_lr.Width(); i++)
			{
				dir1(i - 1, j - 1) = abs(padded_lr(i, j).y - padded_lr(i - 1, j - 1).y);
				dir2(i - 1, j - 1) = abs(padded_lr(i - 1, j).y - padded_lr(i, j - 1).y);
			}
		});

		Parallel::For(0, padded_lr.Height() - 3, [&padded_lr, &dst, &dir1, &dir2, &res, this](int j)
		{
			for (int i = 0; i < padded_lr.Width() - 3; i++)
			{
				// VectorFloat w = CalcWeights(AverageAbnormal3x3_dir1(padded_lr, i + 1, j + 1), AverageNormal3x3(dir2, i + 1, j + 1));
				VectorFloat w = CalcWeights(AverageNormal3x3(dir1, i + 1, j + 1), AverageNormal3x3(dir2, i + 1, j + 1));
				// VectorFloat w = CalcWeights(p1(i, j), p2(i, j));
				VectorFloat dw = ONE - w;

				VectorFloatColor v0 = padded_lr(i, j) + padded_lr(i + 3, j + 3);
				VectorFloatColor v3 = padded_lr(i + 3, j) + padded_lr(i, j + 3);

				VectorFloatColor res0 = (v0 * w + v3 * dw) * kernel1[0] + (v3 * w + v0 * dw) * kernel1[3];

				VectorFloatColor v1 = padded_lr(i + 1, j) + padded_lr(i, j + 1) + padded_lr(i + 3, j + 2) + padded_lr(i + 2, j + 3);
				VectorFloatColor v2 = padded_lr(i + 2, j) + padded_lr(i, j + 2) + padded_lr(i + 3, j + 1) + padded_lr(i + 1, j + 3);

				VectorFloatColor res1 = (v1 * w + v2 * dw) * kernel1[1] + (v2 * w + v1 * dw) * kernel1[2];

				VectorFloatColor v4 = padded_lr(i + 1, j + 1) + padded_lr(i + 2, j + 2);
				VectorFloatColor v5 = padded_lr(i + 1, j + 2) + padded_lr(i + 2, j + 1);

				VectorFloatColor res2 = (v4 * w + v5 * dw) * kernel1[4] + (v5 * w + v4 * dw) * kernel1[5];

				VectorFloatColor c = res0 + res1 + res2;

				dst(i * 2 + 3, j * 2 + 3) = c;
				res(i * 2 + 3, j * 2 + 3) = c.ToByte();
			}
		});
	}

	void EDRVector::Impl::ProcessStep2(VectorImageFloatColor &dst, Image<VectorByte> &res)
	{
		VectorImageFloat dir1(dst.Width() - 9, dst.Height() - 9);
		VectorImageFloat dir2(dst.Width() - 9, dst.Height() - 9);

		Parallel::For(4, dst.Height() - 5, [&dst, &dir1, &dir2](int j)
		{
			for (int i = 5 - (j % 2); i < dst.Width() - 5; i += 2)
			{
				dir1(i - 4, j - 4) = abs(dst(i + 1, j).y - dst(i - 1, j).y);
				dir2(i - 4, j - 4) = abs(dst(i, j + 1).y - dst(i, j - 1).y);
			}
		});

		Parallel::For(6, dst.Height() - 8, [&dst, &dir1, &dir2, &res, this](int j)
		{
			for (int i = 7 - (j % 2); i < dst.Width() - 8; i += 2)
			{
				VectorFloat w = CalcWeights(AverageDiag3x3(dir1, i - 4, j - 4), AverageDiag3x3(dir2, i - 4, j - 4));
				VectorFloat dw = ONE - w;

				VectorFloatColor v0 = dst(i, j - 3) + dst(i, j + 3);
				VectorFloatColor v3 = dst(i - 3, j) + dst(i + 3, j);

				VectorFloatColor res0 = (v0 * w + v3 * dw) * kernel2[0] + (v3 * w + v0 * dw) * kernel2[3];

				VectorFloatColor v1 = dst(i - 1, j - 2) + dst(i + 1, j - 2) + dst(i - 1, j + 2) + dst(i + 1, j + 2);
				VectorFloatColor v2 = dst(i - 2, j - 1) + dst(i + 2, j - 1) + dst(i - 2, j + 1) + dst(i + 2, j + 1);

				VectorFloatColor res1 = (v1 * w + v2 * dw) * kernel2[1] + (v2 * w + v1 * dw) * kernel2[2];

				VectorFloatColor v4 = dst(i, j - 1) + dst(i, j + 1);
				VectorFloatColor v5 = dst(i - 1, j) + dst(i + 1, j);

				VectorFloatColor res2 = (v4 * w + v5 * dw) * kernel2[4] + (v5 * w + v4 * dw) * kernel2[5];

				VectorFloatColor c = res0 + res1 + res2;

				// dst(i, j) = res;
				res(i, j) = c.ToByte();
			}
		});
	}

	inline VectorFloat EDRVector::Impl::CalcWeights(VectorFloat p, VectorFloat q)
	{
		VectorFloat p2 = p * p;
		VectorFloat q2 = q * q;
		VectorFloat p6 = p2 * p2 * p2 + ONE;
		VectorFloat q6 = q2 * q2 * q2 + ONE;
		return p6 / (p6 + q6);
	}

	inline VectorFloat EDRVector::Impl::abs(VectorFloat value)
	{
		return value & SIGNMASK;
	}

	// ==================================================================================================

	#pragma region EDRVector

	EDRVector::EDRVector()
		: impl(new Impl()) {}

	EDRVector::~EDRVector()
	{
		delete impl;
	}

	ImageByteColor EDRVector::Perform(const ImageByteColor &lr)
	{
		BlockSplit bs;
		bs.SetInputPadding(BlockSplitConfiguration(3, 3, 4, 4), BlockSplitPaddingMode::Duplicate);
		bs.SetOutputPadding(BlockSplitConfiguration(6, 6, 8, 8));
		bs.ComputeSplitLayout(lr.Width(), lr.Height());

		 LARGE_INTEGER pFreq, start, cp1, cp2, stop;
		 QueryPerformanceFrequency(&pFreq);
		 QueryPerformanceCounter(&start);

		VectorImageFloatColor src = bs.Split(lr);

		 QueryPerformanceCounter(&cp1);

		Image<VectorByte> dst = impl->Perform(src);

		 QueryPerformanceCounter(&cp2);

		ImageByteColor res = bs.Gather(dst);
		res.SetSize(lr.Width() * 2, lr.Height() * 2);

		 QueryPerformanceCounter(&stop);

		 printf("Prepare: %.3f ms, Run: %.3f ms, Finalize: %.3f ms\n", (cp1.QuadPart - start.QuadPart) * 1e3f / pFreq.QuadPart, (cp2.QuadPart - cp1.QuadPart) * 1e3f / pFreq.QuadPart, (stop.QuadPart - cp2.QuadPart) * 1e3f / pFreq.QuadPart);

		return res;
	}

	void EDRVector::Perform(const VectorImageFloatColor &padded_lr, Image<VectorByte> &dst)
	{
		impl->Perform(padded_lr, dst);
	}

	void EDRVector::LearnStep1(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr)
	{
		impl->LearnStep1(padded_lr, hr);
	}

	void EDRVector::UpdateCoefficientsStep1()
	{
		impl->UpdateCoefficientsStep1();
	}

	void EDRVector::LearnStep2(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr)
	{
		impl->LearnStep2(padded_lr, hr);
	}

	void EDRVector::UpdateCoefficientsStep2()
	{
		impl->UpdateCoefficientsStep2();
	}

	#pragma	endregion

	// ==================================================================================================
}
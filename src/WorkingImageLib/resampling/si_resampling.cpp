#include "si_resampling.h"
#include <iplib/math/quadratic_optimization.h>
#include <misc/blocksplit.h>
#include <vector>

namespace ip
{
	class SIResampling::Impl
	{
	public:
		template <int R, int Q> class SIBase;
		template <int R> class SIResamplingBase;
		template <int R> class SIDeblurBase;

	public:
		virtual Image<VectorByte> Perform(const VectorImageFloatColor &lr) = 0;
		virtual void AddLearningImage(const VectorImageFloatColor &lr, const VectorImageFloatColor &hr) = 0;
		virtual void PerformLearning() = 0;
		virtual size_t SaveCoefficientData(void *buffer, size_t buffer_length) = 0;
		virtual void LoadCoefficientData(void *buffer, size_t buffer_length) = 0;
		virtual void GetLearningStats(int &min, int &aver, int &max) = 0;
		virtual int GetPadding() const = 0;

		static VectorFloat GetBlockIndex(const VectorImageFloatColor &img, int x, int y);
		static VectorInt GetDirectionalIndex(const VectorImageFloatColor &img, int x, int y);

	protected:

		std::vector<std::unique_ptr<QuadraticOptimization[]>> opt;

		void InitLearning(int Q);
	};

	VectorFloat SIResampling::Impl::GetBlockIndex(const VectorImageFloatColor &img, int x, int y)
	{
		static const VectorFloat THR(15.0f * 15.0f);
		static const VectorFloat ONE(1), TWO(2), THREE(3), FOUR(4);
		static const VectorFloat SQ2(0.70710678118654752440084436210485f);

		VectorFloat dx = img(x, y).y + img(x, y + 1).y - img(x + 1, y).y - img(x + 1, y + 1).y;
		VectorFloat dy = img(x, y).y + img(x + 1, y).y - img(x, y + 1).y - img(x + 1, y + 1).y;
		VectorFloat g = dx * dx + dy * dy;
		VectorFloat gt_mask = g.compare_gt_ord_ns(THR);

		VectorFloat v1 = dy.abs();
		VectorFloat v2 = (dx - dy).abs() * SQ2;
		VectorFloat v3 = dx.abs();
		VectorFloat v4 = (dx + dy).abs() * SQ2;

		VectorFloat cmp12 = v1.compare_lt_ord_ns(v2);	// (v1 < v2) ? -1 : 0
		VectorFloat max12 = v1.blend(v2, cmp12);
		VectorFloat res12 = ONE.blend(TWO, cmp12);

		VectorFloat cmp34 = v3.compare_lt_ord_ns(v4);	// (v3 < v4) ? -1 : 0
		VectorFloat max34 = v3.blend(v4, cmp34);
		VectorFloat res34 = THREE.blend(FOUR, cmp34);

		VectorFloat cmp1234 = max12.compare_lt_ord_ns(max34);
		VectorFloat max1234 = max12.blend(max34, cmp1234);
		VectorFloat res1234 = res12.blend(res34, cmp1234);

		return gt_mask & res1234;
	}

	VectorInt SIResampling::Impl::GetDirectionalIndex(const VectorImageFloatColor &img, int x, int y)
	{
		static const VectorFloat FIVE(5.0f);

		VectorFloat i0 = GetBlockIndex(img, x - 1, y - 1);
		VectorFloat i1 = GetBlockIndex(img, x, y - 1);
		VectorFloat i2 = GetBlockIndex(img, x - 1, y);
		VectorFloat i3 = GetBlockIndex(img, x, y);

		return (((i3 * FIVE + i2) * FIVE + i1) * FIVE + i0).conv_i32();
	}

	void SIResampling::Impl::InitLearning(int Q)
	{
		if (opt.size() > 0)
			return;

		for (int i = 0; i < Q; i++)
			opt.emplace_back(std::make_unique<QuadraticOptimization[]>(625));
	}

	// ==================================================================================================

	template <int R, int Q>
	class SIResampling::Impl::SIBase
		: public SIResampling::Impl
	{
	protected:
		static constexpr int R2 = 2 * R + 1;
		float kernels[625][Q][R2 * R2];

	public:

		size_t SaveCoefficientData(void *buffer, size_t buffer_length) override final
		{
			if (buffer_length < sizeof(kernels))
				return sizeof(kernels);

			memcpy(buffer, kernels, sizeof(kernels));

			return sizeof(kernels);
		}

		void LoadCoefficientData(void *buffer, size_t buffer_length) override final
		{
			check(buffer_length == sizeof(kernels));
			memcpy(kernels, buffer, sizeof(kernels));
		}

		int GetPadding() const override
		{
			return R;
		}

		void GetLearningStats(int &min, int &aver, int &max)
		{
			min = max = aver = opt[0][0].Count();

			for (int i = 1; i < 625; i++)
			{
				int N = opt[0][i].Count();
				if (N < min)
					min = N;
				if (N > max)
					max = N;
				aver += N;
			}

			aver /= 625;
		}

		void PerformLearning() override final
		{
			for (int i = 0; i < 625; i++)
				for (int j = 0; j < Q; j++)
					opt[j][i].CalcOptimizedCoefficients(kernels[i][j]);
		}
	};

	// ==================================================================================================

	template <int R>
	class SIResampling::Impl::SIResamplingBase
		: public SIResampling::Impl::SIBase<R, 4>
	{
		static constexpr int R2 = SIResampling::Impl::SIBase<R, 4>::R2;

		Image<VectorByte> Perform(const VectorImageFloatColor &lr) override final
		{
			Image<VectorByte> hr{ (lr.Width() - 2 * R) * 2, (lr.Height() - 2 * R) * 2 };

			Parallel::For(0, lr.Height() - 2 * R, [&lr, &hr, this](int j)
			{
				for (int i = 0; i < lr.Width() - 2 * R; i++)
				{
					VectorInt indices = GetDirectionalIndex(lr, i + R, j + R);

					VectorFloatColor x[R2 * R2];
					for (int jj = 0; jj < R2; jj++)
						for (int ii = 0; ii < R2; ii++)
							x[jj * R2 + ii] = lr(i + ii, j + jj);

					VectorFloatColor z[4];

					for (int n = 0; n < 4; n++)
					{
						for (int k = 0; k < VectorFloat::size; k++)
						{
							float y = 0.0f, u = 0.0f, v = 0.0f;
							int idx = indices.get(k);

							for (int p = 0; p < R2 * R2; p++)
							{
								y += this->kernels[idx][n][p] * x[p].y.get(k);
								u += this->kernels[idx][n][p] * x[p].u.get(k);
								v += this->kernels[idx][n][p] * x[p].v.get(k);
							}

							z[n].y.set(k, y);
							z[n].u.set(k, u);
							z[n].v.set(k, v);
						}
					}

					hr(i * 2, j * 2) = z[0].ToByte();
					hr(i * 2 + 1, j * 2) = z[1].ToByte();
					hr(i * 2, j * 2 + 1) = z[2].ToByte();
					hr(i * 2 + 1, j * 2 + 1) = z[3].ToByte();
				}
			});

			return hr;
		}

		void AddLearningImage(const VectorImageFloatColor &lr, const VectorImageFloatColor &hr) override final
		{
			this->InitLearning(4);

			for (int j = 3; j < lr.Height() - 3; j++)
				for (int i = 3; i < lr.Width() - 3; i++)
				{
					VectorInt indices = GetDirectionalIndex(lr, i, j);

					VectorFloat vkernel[R2 * R2];

					for (int jj = 0; jj < R2; jj++)
						for (int ii = 0; ii < R2; ii++)
							vkernel[jj * R2 + ii] = lr(i + ii - R, j + jj - R).y;

					VectorFloat vv[4] = { hr(2 * i, 2 * j).y, hr(2 * i + 1, 2 * j).y, hr(2 * i, 2 * j + 1).y, hr(2 * i + 1, 2 * j + 1).y };

					for (int k = 0; k < VectorFloat::size; k++)
					{
						float kernel[R2 * R2];

						for (int p = 0; p < R2 * R2; p++)
							kernel[p] = vkernel[p].get(k);

						int idx = indices.get(k);

						this->opt[0][idx].AddData(kernel, vv[0].get(k));
						this->opt[1][idx].AddData(kernel, vv[1].get(k));
						this->opt[2][idx].AddData(kernel, vv[2].get(k));
						this->opt[3][idx].AddData(kernel, vv[3].get(k));
					}
				}
		}






	};

	// ==================================================================================================

	template <int R>
	class SIResampling::Impl::SIDeblurBase
		: public SIResampling::Impl::SIBase<R, 1>
	{
		static constexpr int R2 = SIResampling::Impl::SIBase<R, 1>::R2;

		Image<VectorByte> Perform(const VectorImageFloatColor &lr) override final
		{
			Image<VectorByte> hr{ lr.Width() - 2 * R, lr.Height() - 2 * R };

			Parallel::For(0, lr.Height() - 2 * R, [&lr, &hr, this](int j)
			{
				for (int i = 0; i < lr.Width() - 2 * R; i++)
				{
					VectorInt indices = GetDirectionalIndex(lr, i + R, j + R);

					VectorFloatColor x[R2 * R2];
					for (int jj = 0; jj < R2; jj++)
						for (int ii = 0; ii < R2; ii++)
							x[jj * R2 + ii] = lr(i + ii, j + jj);

					VectorFloatColor z;

					for (int k = 0; k < VectorFloat::size; k++)
					{
						float y = 0.0f, u = 0.0f, v = 0.0f;
						int idx = indices.get(k);

						for (int p = 0; p < R2 * R2; p++)
						{
							y += this->kernels[idx][0][p] * x[p].y.get(k);
							u += this->kernels[idx][0][p] * x[p].u.get(k);
							v += this->kernels[idx][0][p] * x[p].v.get(k);
						}

						z.y.set(k, y);
						z.u.set(k, u);
						z.v.set(k, v);
					}

					hr(i, j) = z.ToByte();
				}
			});

			return hr;
		}

		void AddLearningImage(const VectorImageFloatColor &lr, const VectorImageFloatColor &hr) override final
		{
			this->InitLearning(1);

			for (int j = 3; j < lr.Height() - 3; j++)
				for (int i = 3; i < lr.Width() - 3; i++)
				{
					VectorInt indices = GetDirectionalIndex(lr, i, j);

					VectorFloat vkernel[R2 * R2];

					for (int jj = 0; jj < R2; jj++)
						for (int ii = 0; ii < R2; ii++)
							vkernel[jj * R2 + ii] = lr(i + ii - R, j + jj - R).y;

					VectorFloat vv = hr(i, j).y;

					for (int k = 0; k < VectorFloat::size; k++)
					{
						float kernel[R2 * R2];

						for (int p = 0; p < R2 * R2; p++)
							kernel[p] = vkernel[p].get(k);

						int idx = indices.get(k);

						this->opt[0][idx].AddData(kernel, vv.get(k));
					}
				}
		}
	};

	// ==================================================================================================

	SIResampling::SIResampling(Mode mode, void *coefficient_data, size_t coefficient_data_size)
	{
		switch (mode)
		{
		case Mode::SI1:
			this->impl = new SIResampling::Impl::SIResamplingBase<1>();
			break;

		case Mode::SI2:
			this->impl = new SIResampling::Impl::SIResamplingBase<2>();
			break;

		case Mode::SI3:
			this->impl = new SIResampling::Impl::SIResamplingBase<3>();
			break;

		case Mode::SI1Deblur:
			this->impl = new SIResampling::Impl::SIDeblurBase<1>();
			break;

		case Mode::SI2Deblur:
			this->impl = new SIResampling::Impl::SIDeblurBase<2>();
			break;

		case Mode::SI3Deblur:
			this->impl = new SIResampling::Impl::SIDeblurBase<3>();
			break;

		default:
			die("Invalid mode");
		}

		if (coefficient_data != nullptr)
		{
			impl->LoadCoefficientData(coefficient_data, coefficient_data_size);
		}
	}

	SIResampling::SIResampling(Mode mode)
		: SIResampling(mode, nullptr, 0) {}

	SIResampling::~SIResampling()
	{
		delete this->impl;
	}

	ImageByteColor SIResampling::Perform(const ImageByteColor &lr)
	{
		BlockSplit bs;
		bs.SetInputPadding(BlockSplitConfiguration(impl->GetPadding()), BlockSplitPaddingMode::Duplicate);
		bs.ComputeSplitLayout(lr.Width(), lr.Height());

		VectorImageFloatColor vlr = bs.Split(lr);

		Image<VectorByte> tmp = impl->Perform(vlr);

		ImageByteColor res = bs.Gather(tmp);
		res.SetSize(lr.Width() * 2, lr.Height() * 2);
		return res;
	}

	ImageByteColor SIResampling::PerformDeblur(const ImageByteColor &lr)
	{
		BlockSplit bs;
		bs.SetInputPadding(BlockSplitConfiguration(impl->GetPadding()), BlockSplitPaddingMode::Duplicate);
		bs.ComputeSplitLayout(lr.Width(), lr.Height());

		VectorImageFloatColor vlr = bs.Split(lr);

		Image<VectorByte> tmp = impl->Perform(vlr);

		ImageByteColor res = bs.Gather(tmp);
		res.SetSize(lr.Width(), lr.Height());
		return res;
	}

	void SIResampling::AddLearningImage(const ImageByteColor &lr, const ImageByteColor &hr)
	{
		check(lr.Width() * 2 == hr.Width() && lr.Height() * 2 == hr.Height());

		BlockSplit bs;
		bs.ComputeSplitLayout(lr.Width(), lr.Height());

		VectorImageFloatColor vlr = bs.Split(lr);
		VectorImageFloatColor vhr = bs.Split(hr, vlr.Width() * 2, vlr.Height() * 2);

		impl->AddLearningImage(vlr, vhr);
	}

	void SIResampling::AddLearningImageDeblur(const ImageByteColor &lr, const ImageByteColor &hr)
	{
		check(lr.Width() == hr.Width() && lr.Height() == hr.Height());

		BlockSplit bs;
		bs.ComputeSplitLayout(lr.Width(), lr.Height());

		VectorImageFloatColor vlr = bs.Split(lr);
		VectorImageFloatColor vhr = bs.Split(hr, vlr.Width(), vlr.Height());

		impl->AddLearningImage(vlr, vhr);
	}

	void SIResampling::GetLearningStats(int &min, int &aver, int &max)
	{
		impl->GetLearningStats(min, aver, max);
	}

	void SIResampling::PerformLearning()
	{
		impl->PerformLearning();
	}

	size_t SIResampling::SaveCoefficientData(void *buffer, size_t buffer_length)
	{
		return impl->SaveCoefficientData(buffer, buffer_length);
	}
}
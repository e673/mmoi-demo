#pragma once

#include <iplib/image/core.h>
#include <iplib/math/quadratic_optimization.h>

namespace ip
{
	// ==================================================================================================
	//                                       EDResampling               
	// ==================================================================================================

	class EDResampling
	{
	public:
		struct Kernels
		{
			float kernel0[6];
			float kernel1[6];
			float kernel2[6];	// Isoptropic kernel for step2

			float kernel2a[12];	// Anisotropic kernel for step2 (Anisotropic - Even)
			float kernel2b[12];	// Anisotropic kernel for step2 (Anisotropic - Odd)

			float kernel2c[12];	// Anisotropic kernel for step2 (Anisoptropic2 - Universal)

			void LoadDefault();
		};

		enum class Mode
		{
			Isotropic = 0,
			Anisotropic = 1,
			Anisoptropic2 = 2
		};

	public:

		EDResampling();
		EDResampling(Kernels kernels);

		void Configure(bool fast_weights);

		template <class PixelType, class SourceImageType, class DestinationImageType>
		void Perform(const ImageReadable<PixelType, SourceImageType> & src, ImageWritable<PixelType, DestinationImageType> & dst, EDResampling::Mode mode);

	protected:
		Kernels kernels;
		bool fast_weights = false;

	protected:
		template <class PixelType, class SourceImageType> class SourceImage;
		template <class PixelType, class DestinationImageType> class DestinationImage;

		template <class PixelType> class CoefficientsStep0;
		template <class PixelType> class CoefficientsStep1;
		template <class PixelType> class CoefficientsStep2;

		template <class PixelType, class ImageType>	class SafeWrapImpl;

		class WeightCalculator;

		template <class PixelType, class SourcePixelType, class DestinationPixelType> class Resampler;
		template <class PixelType, class SourcePixelType, class DestinationPixelType> class Learner;

	private:
		static void Average3x3(const Image<float> &src, Image<float> &dst);
		static void Average3x3diag(const Image<float> &src, Image<float> &dst);

		static float CalcWeight(float g1, float g2);
		static float CalcWeightFast(float g1, float g2);

		static constexpr float Threshold = 1.25f;

		template<class ImageType>
		auto static SafeWrap(const ImageType &img) -> SafeWrapImpl<decltype(img(0, 0)), ImageType>;
	};

	// ==================================================================================================
	//                                     EDResamplingExt               
	// ==================================================================================================

	class EDResamplingExt
		: public EDResampling
	{
		using EDResampling::EDResampling;

	public:
		template <class PixelType, class SourceImageType, class DestinationImageType>
		void LearnStep1(const ImageReadable<PixelType, SourceImageType> & lr, const ImageReadable<PixelType, DestinationImageType> & hr);

		template <class PixelType, class SourceImageType, class DestinationImageType>
		void LearnStep2(const ImageReadable<PixelType, SourceImageType> & lr, const ImageReadable<PixelType, DestinationImageType> & hr);

		void UpdateCoefficients();
		void UpdateCoefficientsStep1();
		void UpdateCoefficientsStep2();

		EDResampling::Kernels GetKernels() const;

	private:
		QuadraticOptimization opt0, opt1, opt2, opt2a, opt2b, opt2c;
	};

	// ==================================================================================================
	//                                 SourceImage
	// ==================================================================================================

	template <class PixelType, class SourceImageType>
	class EDResampling::SourceImage
	{
		const ip::ImageReadable<PixelType, SourceImageType> &img;
		ip::Image<float> imgf;

	public:
		SourceImage(const ip::ImageReadable<PixelType, SourceImageType> &img)
			: img(img), imgf(img.Width(), img.Height())
		{
			ip::Parallel::For(0, img.Height(), [&img, this](int j)
			{
				for (int i = 0; i < img.Width(); i++)
				{
					imgf(i, j) = (float)img(i, j);
				}
			});
		}

		PixelType operator()(int x, int y) const
		{
			return img.pixel(x, y);
		}

		float getf(int x, int y) const
		{
			return imgf.pixel(x, y);
		}

		int Width() const
		{
			return img.Width();
		}

		int Height() const
		{
			return img.Height();
		}

		const Image<float> & get_imgf() const
		{
			return imgf;
		}
	};

	template <class SourceImageType>
	class EDResampling::SourceImage<float, SourceImageType>
	{
		const ip::ImageReadable<float, SourceImageType> &img;

	public:
		SourceImage(const ip::ImageReadable<float, SourceImageType> &img)
			: img(img) {}

		float operator()(int x, int y) const
		{
			return img.pixel(x, y);
		}

		float getf(int x, int y) const
		{
			return img.pixel(x, y);
		}

		int Width() const
		{
			return img.Width();
		}

		int Height() const
		{
			return img.Height();
		}

		const ip::ImageReadable<float, SourceImageType> & get_imgf() const
		{
			return img;
		}
	};

	// ==================================================================================================
	//                               DestinationImage
	// ==================================================================================================

	template <class PixelType, class DestinationImageType>
	class EDResampling::DestinationImage
	{
		ip::ImageWritable<PixelType, DestinationImageType> &img;
		ip::Image<float> imgf;

	public:
		DestinationImage(ip::ImageWritable<PixelType, DestinationImageType> &img)
			: img(img), imgf(img.Width(), img.Height()) {}

		PixelType operator()(int x, int y) const
		{
			return img.pixel(x, y);
		}

		float getf(int x, int y) const
		{
			return imgf.pixel(x, y);
		}

		void set(int x, int y, PixelType v)
		{
			img(x, y) = v;
			imgf(x, y) = (float)v;
		}

		int Width() const
		{
			return img.Width();
		}

		int Height() const
		{
			return img.Height();
		}

		const ip::Image<float> & get_imgf() const
		{
			return imgf;
		}
	};

	template <class DestinationImageType>
	class EDResampling::DestinationImage<float, DestinationImageType>
	{
		ip::ImageWritable<float, DestinationImageType> &img;

	public:
		DestinationImage(ip::ImageWritable<float, DestinationImageType> &img)
			: img(img) {}

		float operator()(int x, int y) const
		{
			return img.pixel(x, y);
		}

		float getf(int x, int y) const
		{
			return img.pixel(x, y);
		}

		void set(int x, int y, float v)
		{
			img(x, y) = v;
		}

		int Width() const
		{
			return img.Width();
		}

		int Height() const
		{
			return img.Height();
		}

		const ip::ImageReadable<float, DestinationImageType> & get_imgf() const
		{
			return img;
		}
	};

	// ==================================================================================================
	//                               CoefficientsStep
	// ==================================================================================================

	template <class PixelType>
	class EDResampling::CoefficientsStep0
	{
	public:
		PixelType data[6];

		PixelType ApplyKernel(const float * kernel) const
		{
			return data[0] * kernel[0] + data[1] * kernel[1] + data[2] * kernel[2] + data[3] * kernel[3] + data[4] * kernel[4] + data[5] * kernel[5];
		}
	};

	template <class PixelType>
	class EDResampling::CoefficientsStep1
	{
	public:
		PixelType data[6];

		PixelType ApplyKernel(const float * kernel, float w) const
		{
			float w1 = 1.0f - w;

			return (data[0] * w + data[3] * w1) * kernel[0] +
				   (data[1] * w + data[2] * w1) * kernel[1] +
				   (data[2] * w + data[1] * w1) * kernel[2] +
				   (data[3] * w + data[0] * w1) * kernel[3] +
				   (data[4] * w + data[5] * w1) * kernel[4] +
				   (data[5] * w + data[4] * w1) * kernel[5];
		}
	};

	template <class PixelType>
	class EDResampling::CoefficientsStep2
	{
	public:
		PixelType data[6];

		PixelType ApplyKernel(const float * kernel, float w) const
		{
			float w1 = 1.0f - w;

			return (data[0] * w + data[3] * w1) * kernel[0] +
				   (data[1] * w + data[2] * w1) * kernel[1] +
				   (data[2] * w + data[1] * w1) * kernel[2] +
				   (data[3] * w + data[0] * w1) * kernel[3] +
				   (data[4] * w + data[5] * w1) * kernel[4] +
				   (data[5] * w + data[4] * w1) * kernel[5];
		}

		PixelType ApplyKernelAnisotropic(const float * kernel, float w) const
		{
			float w1 = 1.0f - w;

			return data[0] * (w * kernel[0] + w1 * kernel[6]) +
				   data[1] * (w * kernel[1] + w1 * kernel[7]) +
				   data[2] * (w * kernel[2] + w1 * kernel[8]) +
				   data[3] * (w * kernel[3] + w1 * kernel[9]) +
				   data[4] * (w * kernel[4] + w1 * kernel[10]) +
				   data[5] * (w * kernel[5] + w1 * kernel[11]);
		}

		PixelType ApplyKernelAnisotropicEven(const float * kernel, float w) const
		{
			float w1 = 1.0f - w;

			return data[0] * (w * kernel[0] + w1 * kernel[6]) +
				   data[1] * (w * kernel[1] + w1 * kernel[7]) +
				   data[2] * (w * kernel[2] + w1 * kernel[8]) +
				   data[3] * (w * kernel[3] + w1 * kernel[9]) +
				   data[4] * (w * kernel[4] + w1 * kernel[10]) +
				   data[5] * (w * kernel[5] + w1 * kernel[11]);
		}

		PixelType ApplyKernelAnisotropicOdd(const float * kernel, float w) const
		{
			float w1 = 1.0f - w;

			return data[3] * (w1 * kernel[0] + w * kernel[6]) +
				   data[2] * (w1 * kernel[1] + w * kernel[7]) +
				   data[1] * (w1 * kernel[2] + w * kernel[8]) +
				   data[0] * (w1 * kernel[3] + w * kernel[9]) +
				   data[5] * (w1 * kernel[4] + w * kernel[10]) +
				   data[4] * (w1 * kernel[5] + w * kernel[11]);
		}
	};

	// ==================================================================================================
	//                                 SafeWrapImpl
	// ==================================================================================================

	template <class PixelType, class ImageType>
	class EDResampling::SafeWrapImpl
	{
		const ImageType &img;

	public:
		SafeWrapImpl(const ImageType &img)
			: img(img) {}

		inline int Width() const
		{
			return img.Width();
		}

		inline int Height() const
		{
			return img.Height();
		}

		inline PixelType operator()(int x, int y) const
		{
			if (x < 0)
				x = -x;

			if (x >= img.Width())
				x = 2 * img.Width() - 2 - x;

			if (y < 0)
				y = -y;

			if (y >= img.Height())
				y = 2 * img.Height() - 2 - y;

			return img(x, y);
		}
	};

	// ==================================================================================================
	//                               WeightCalculator
	// ==================================================================================================

	class EDResampling::WeightCalculator
	{
		Image<float> data;
		bool fast;

	public:
		WeightCalculator(bool fast, int Width, int Height);

		float get(int x, int y) const;

		template <class PixelType>
		void CalcStep0(const ImageReadable<float, PixelType> &img)
		{
			dbgcheck((img.Width() * 2 == data.Width()) && (img.Height() * 2 == data.Height()));
			// ToDo
		}

		template <class PixelType>
		void CalcStep1(const ImageReadable<float, PixelType> &img)
		{
			dbgcheck((img.Width() * 2 == data.Width()) && (img.Height() * 2 == data.Height()));

			Image<float> dir1(img.Width(), img.Height());
			Image<float> dir2(img.Width(), img.Height());
			Image<float> dir1a(img.Width(), img.Height());
			Image<float> dir2a(img.Width(), img.Height());

			Parallel::For(0, img.Height() - 1, [&dir1, &dir2, &dir1a, &dir2a, &img](int j)
			{
				for (int i = 0; i < img.Width() - 1; i++)
				{
					dir1(i, j) = fabsf(img(i + 1, j) - img(i, j + 1));
					dir2(i, j) = fabsf(img(i + 1, j + 1) - img(i, j));
				}

				dir1(img.Width() - 1, j) = dir2(img.Width() - 1, j) = 0.0f;
			});

			for (int i = 0; i < img.Width(); i++)
			{
				dir1(i, img.Height() - 1) = dir2(i, img.Height() - 1) = 0.0f;
			}

			Average3x3(dir1, dir1a);
			Average3x3(dir2, dir2a);

			if (fast)
			{
				Parallel::For(0, img.Height(), [&dir1a, &dir2a, &img, this](int j)
				{
					for (int i = 0; i < img.Width(); i++)
						data(i * 2 + 1, j * 2 + 1) = CalcWeightFast(dir1a(i, j), dir2a(i, j));
				});
			}
			else
			{
				Parallel::For(0, img.Height(), [&dir1a, &dir2a, &img, this](int j)
				{
					for (int i = 0; i < img.Width(); i++)
						data(i * 2 + 1, j * 2 + 1) = CalcWeight(dir1a(i, j), dir2a(i, j));
				});
			}
		}

		template <class PixelType>
		void CalcStep2(const ImageReadable<float, PixelType> &img)
		{
			dbgcheck((img.Width() == data.Width()) && (img.Height() == data.Height()));

			Image<float> dir1(img.Width(), img.Height());
			Image<float> dir2(img.Width(), img.Height());
			Image<float> dir1a(img.Width(), img.Height());
			Image<float> dir2a(img.Width(), img.Height());

			Parallel::For(1, img.Height() - 1, [&dir1, &dir2, &dir1a, &dir2a, &img](int j)
			{
				int ofs = 1 - (j % 2);

				dir1(ofs, j) = dir2(ofs, j) = 0.0f;
				dir1(img.Width() - 2 + ofs, j) = dir2(img.Width() - 2 + ofs, j) = 0.0f;

				for (int i = 2 - ofs; i < img.Width() - 1; i += 2)
				{
					dir1(i, j) = fabsf(img(i, j + 1) - img(i, j - 1));
					dir2(i, j) = fabsf(img(i + 1, j) - img(i - 1, j));
				}
			});

			for (int i = 1; i < img.Width(); i += 2)
			{
				dir1(i, 0) = dir2(i, 0) = 0.0f;
				dir1(i - 1, img.Height() - 1) = dir2(i - 1, img.Height() - 1) = 0.0f;
			}

			Average3x3diag(dir1, dir1a);
			Average3x3diag(dir2, dir2a);

			if (fast)
			{
				Parallel::For(0, img.Height(), [&dir1a, &dir2a, &img, this](int j)
				{
					for (int i = 1 - (j % 2); i < img.Width(); i += 2)
					{
						data(i, j) = CalcWeightFast(dir1a(i, j), dir2a(i, j));
					}
				});
			}
			else
			{
				Parallel::For(0, img.Height(), [&dir1a, &dir2a, &img, this](int j)
				{
					for (int i = 1 - (j % 2); i < img.Width(); i += 2)
					{
						data(i, j) = CalcWeight(dir1a(i, j), dir2a(i, j));
					}
				});
			}
		}
	};

	// ==================================================================================================
	//                                 Resampler
	// ==================================================================================================

	template <class PixelType, class SourcePixelType, class DestinationPixelType>
	class EDResampling::Resampler
	{
	protected:
		EDResampling::SourceImage<PixelType, SourcePixelType> src;
		EDResampling::DestinationImage<PixelType, DestinationPixelType> dst;
		EDResampling::Kernels kernels;
		EDResampling::WeightCalculator weights;

	public:
		Resampler(const ImageReadable<PixelType, SourcePixelType> &src, ImageWritable<PixelType, DestinationPixelType> &dst, EDResampling::Kernels kernels, bool fast);

		void Perform(EDResampling::Mode mode);

	protected:
		void PerformStep0();
		void PerformStep1();
		void PerformStep2(EDResampling::Mode mode);

		template <class T>
		auto GetCoefficientsStep0(const T &t, int x, int y) const;

		template <class T>
		auto GetCoefficientsStep1(const T &t, int x, int y) const;

		template <class T>
		auto GetCoefficientsStep2(const T &t, int x, int y) const;

		template <class T>
		PixelType IterateStep0(const T &t, int x, int y) const;

		template <class T>
		PixelType IterateStep1(const T &t, int x, int y) const;

		template <class T>
		PixelType IterateStep2(const T &t, int x, int y) const;

		template <class T>
		PixelType IterateStep2a(const T &t, int x, int y) const;

		template <class T>
		PixelType IterateStep2b(const T &t, int x, int y) const;
	};

	// ==================================================================================================

	template <class PixelType, class SourcePixelType, class DestinationPixelType>
	EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::Resampler(const ImageReadable<PixelType, SourcePixelType> &src, ImageWritable<PixelType, DestinationPixelType> &dst, EDResampling::Kernels kernels, bool fast)
		: src(src), dst(dst), kernels(kernels), weights(fast, src.Width() * 2, src.Height() * 2)
	{
		check(src.Width() * 2 == dst.Width() && src.Height() * 2 == dst.Height());
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	void EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::Perform(Mode mode)
	{
		PerformStep0();
		PerformStep1();
		PerformStep2(mode);
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline auto EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::GetCoefficientsStep0(const T &t, int x, int y) const
		// -> EDResampling::CoefficientsStep0<decltype(t(0, 0))>
	{
		EDResampling::CoefficientsStep0<decltype(t(0, 0))> res;

		res.data[0] = t(x - 2, y - 2) + t(x + 2, y - 2) + t(x - 2, y + 2) + t(x + 2, y + 2);
		res.data[1] = t(x - 1, y - 2) + t(x + 1, y - 2) + t(x - 2, y - 1) + t(x + 2, y - 1) +
			t(x - 2, y + 1) + t(x + 2, y + 1) + t(x - 1, y + 2) + t(x + 1, y + 2);
		res.data[2] = t(x, y - 2) + t(x - 2, y) + t(x + 2, y) + t(x, y + 2);
		res.data[3] = t(x - 1, y - 1) + t(x + 1, y - 1) + t(x - 1, y + 1) + t(x + 1, y + 1);
		res.data[4] = t(x, y - 1) + t(x - 1, y) + t(x + 1, y) + t(x, y + 1);
		res.data[5] = t(x, y);

		return res;
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline auto EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::GetCoefficientsStep1(const T &t, int x, int y) const
	{
		EDResampling::CoefficientsStep1<decltype(t(0, 0))> res;

		res.data[0] = t(x - 1, y - 1) + t(x + 2, y + 2);
		res.data[1] = t(x, y - 1) + t(x - 1, y) + t(x + 2, y + 1) + t(x + 1, y + 2);
		res.data[2] = t(x + 1, y - 1) + t(x - 1, y + 1) + t(x + 2, y) + t(x, y + 2);
		res.data[3] = t(x + 2, y - 1) + t(x - 1, y + 2);
		res.data[4] = t(x, y) + t(x + 1, y + 1);
		res.data[5] = t(x + 1, y) + t(x, y + 1);

		return res;
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline auto EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::GetCoefficientsStep2(const T &t, int x, int y) const
	{
		EDResampling::CoefficientsStep2<decltype(t(0, 0))> res;

		res.data[0] = t(x, y - 3) + t(x, y + 3);
		res.data[1] = t(x - 1, y - 2) + t(x + 1, y - 2) + t(x - 1, y + 2) + t(x + 1, y + 2);
		res.data[2] = t(x - 2, y - 1) + t(x + 2, y - 1) + t(x - 2, y + 1) + t(x + 2, y + 1);
		res.data[3] = t(x - 3, y) + t(x + 3, y);
		res.data[4] = t(x, y - 1) + t(x, y + 1);
		res.data[5] = t(x - 1, y) + t(x + 1, y);

		return res;
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline PixelType EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::IterateStep0(const T & src, int x, int y) const
	{
		return GetCoefficientsStep0(src, x, y).ApplyKernel(kernels.kernel0);
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline PixelType EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::IterateStep1(const T & src, int x, int y) const
	{
		return GetCoefficientsStep1(src, x, y).ApplyKernel(kernels.kernel1, weights.get(x * 2 + 1, y * 2 + 1));
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline PixelType EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::IterateStep2(const T & src, int x, int y) const
	{
		return GetCoefficientsStep2(src, x, y).ApplyKernel(kernels.kernel2, weights.get(x, y));
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline PixelType EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::IterateStep2a(const T & src, int x, int y) const
	{
		if (y % 2 == 0)
			return GetCoefficientsStep2(src, x, y).ApplyKernelAnisotropic(kernels.kernel2a, weights.get(x, y));
		else
			return GetCoefficientsStep2(src, x, y).ApplyKernelAnisotropic(kernels.kernel2b, weights.get(x, y));
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class T>
	inline PixelType EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::IterateStep2b(const T & src, int x, int y) const
	{
		if (y % 2 == 0)
			return GetCoefficientsStep2(src, x, y).ApplyKernelAnisotropicEven(kernels.kernel2c, weights.get(x, y));
		else
			return GetCoefficientsStep2(src, x, y).ApplyKernelAnisotropicOdd(kernels.kernel2c, weights.get(x, y));
	}


	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	void EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::PerformStep0()
	{
		weights.CalcStep0(src.get_imgf());

		Parallel::For(0, src.Height(), [this](int j)
		{
			if (j >= 2 && j < src.Height() - 2)
			{
				dst.set(0, j * 2, IterateStep0(SafeWrap(src), 0, j));
				dst.set(2, j * 2, IterateStep0(SafeWrap(src), 1, j));

				for (int i = 2; i < src.Width() - 2; i++)
				{
					dst.set(i * 2, j * 2, IterateStep0(src, i, j));
				}

				dst.set(src.Width() * 2 - 4, j * 2, IterateStep0(SafeWrap(src), src.Width() - 2, j));
				dst.set(src.Width() * 2 - 2, j * 2, IterateStep0(SafeWrap(src), src.Width() - 1, j));
			}
			else
			{
				for (int i = 0; i < src.Width(); i++)
				{
					dst.set(i * 2, j * 2, IterateStep0(SafeWrap(src), i, j));
				}
			}
		});
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	void EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::PerformStep1()
	{
		weights.CalcStep1(src.get_imgf());

		Parallel::For(0, src.Height(), [this](int j)
		{
			if (j >= 1 && j < src.Height() - 2)
			{
				dst.set(1, j * 2 + 1, IterateStep1(SafeWrap(src), 0, j));

				for (int i = 1; i < src.Width() - 2; i++)
				{
					dst.set(i * 2 + 1, j * 2 + 1, IterateStep1(src, i, j));
				}

				dst.set(src.Width() * 2 - 3, j * 2 + 1, IterateStep1(SafeWrap(src), src.Width() - 2, j));
				dst.set(src.Width() * 2 - 1, j * 2 + 1, IterateStep1(SafeWrap(src), src.Width() - 1, j));
			}
			else
			{
				for (int i = 0; i < src.Width(); i++)
				{
					dst.set(i * 2 + 1, j * 2 + 1, IterateStep1(SafeWrap(src), i, j));
				}
			}
		});
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	void EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>::PerformStep2(EDResampling::Mode mode)
	{
		weights.CalcStep2(dst.get_imgf());

		switch (mode)
		{
		case EDResampling::Mode::Isotropic:
			Parallel::For(0, dst.Height(), [this](int j)
			{
				int ix = 1 - (j % 2);

				if (j >= 3 && j < dst.Height() - 3)
				{
					dst.set(ix, j, IterateStep2(SafeWrap(dst), ix, j));
					dst.set(ix + 2, j, IterateStep2(SafeWrap(dst), ix + 2, j));

					for (int i = ix + 4; i < dst.Width() - 4; i += 2)
					{
						dst.set(i, j, IterateStep2(dst, i, j));
					}

					dst.set(dst.Width() - 4 + ix, j, IterateStep2(SafeWrap(dst), dst.Width() - 4 + ix, j));
					dst.set(dst.Width() - 2 + ix, j, IterateStep2(SafeWrap(dst), dst.Width() - 2 + ix, j));
				}
				else
				{
					for (int i = ix; i < dst.Width(); i += 2)
					{
						dst.set(i, j, IterateStep2(SafeWrap(dst), i, j));
					}
				}
			});
			break;

		case EDResampling::Mode::Anisotropic:
			Parallel::For(0, dst.Height(), [this](int j)
			{
				int ix = 1 - (j % 2);

				if (j >= 3 && j < dst.Height() - 3)
				{
					dst.set(ix, j, IterateStep2a(SafeWrap(dst), ix, j));
					dst.set(ix + 2, j, IterateStep2a(SafeWrap(dst), ix + 2, j));

					for (int i = ix + 4; i < dst.Width() - 4; i += 2)
					{
						dst.set(i, j, IterateStep2a(dst, i, j));
					}

					dst.set(dst.Width() - 4 + ix, j, IterateStep2a(SafeWrap(dst), dst.Width() - 4 + ix, j));
					dst.set(dst.Width() - 2 + ix, j, IterateStep2a(SafeWrap(dst), dst.Width() - 2 + ix, j));
				}
				else
				{
					for (int i = ix; i < dst.Width(); i += 2)
					{
						dst.set(i, j, IterateStep2a(SafeWrap(dst), i, j));
					}
				}
			});
			break;

		case EDResampling::Mode::Anisoptropic2:
			Parallel::For(0, dst.Height(), [this](int j)
			{
				int ix = 1 - (j % 2);

				if (j >= 3 && j < dst.Height() - 3)
				{
					dst.set(ix, j, IterateStep2b(SafeWrap(dst), ix, j));
					dst.set(ix + 2, j, IterateStep2b(SafeWrap(dst), ix + 2, j));

					for (int i = ix + 4; i < dst.Width() - 4; i += 2)
					{
						dst.set(i, j, IterateStep2b(dst, i, j));
					}

					dst.set(dst.Width() - 4 + ix, j, IterateStep2b(SafeWrap(dst), dst.Width() - 4 + ix, j));
					dst.set(dst.Width() - 2 + ix, j, IterateStep2b(SafeWrap(dst), dst.Width() - 2 + ix, j));
				}
				else
				{
					for (int i = ix; i < dst.Width(); i += 2)
					{
						dst.set(i, j, IterateStep2b(SafeWrap(dst), i, j));
					}
				}
			});
			break;
		}
	}


	// ==================================================================================================
	//                                  Learner
	// ==================================================================================================

	template <class PixelType, class SourcePixelType, class DestinationPixelType>
	class EDResampling::Learner
		: protected EDResampling::Resampler<PixelType, SourcePixelType, DestinationPixelType>
	{
	public:
		Learner(const ImageReadable<PixelType, SourcePixelType> &src, ImageWritable<PixelType, DestinationPixelType> &dst, EDResampling::Kernels kernels, bool fast);

		void InterpolateStep0();
		void InterpolateStep1();
		void InterpolateStep2();

		template <class ReferencePixelType>
		void LearnStep0(const ImageReadable<PixelType, ReferencePixelType> &reference, QuadraticOptimization &opt);

		template <class ReferencePixelType>
		void LearnStep1(const ImageReadable<PixelType, ReferencePixelType> &reference, QuadraticOptimization &opt);

		template <class ReferencePixelType>
		void LearnStep2(const ImageReadable<PixelType, ReferencePixelType> &reference, QuadraticOptimization &opt2, QuadraticOptimization &opt2a, QuadraticOptimization &opt2b, QuadraticOptimization &opt2c);
	};

	// ==================================================================================================

	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	inline EDResampling::Learner<PixelType, SourcePixelType, DestinationPixelType>::Learner(const ImageReadable<PixelType, SourcePixelType>& src, ImageWritable<PixelType, DestinationPixelType>& dst, EDResampling::Kernels kernels, bool fast)
		: Resampler<PixelType, SourcePixelType, DestinationPixelType>(src, dst, kernels, fast) {}

	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	void EDResampling::Learner<PixelType, SourcePixelType, DestinationPixelType>::InterpolateStep0()
	{
		this->PerformStep0();
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	void EDResampling::Learner<PixelType, SourcePixelType, DestinationPixelType>::InterpolateStep1()
	{
		this->PerformStep1();
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType>
	void EDResampling::Learner<PixelType, SourcePixelType, DestinationPixelType>::InterpolateStep2()
	{
		this->PerformStep2();
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class ReferencePixelType>
	void EDResampling::Learner<PixelType, SourcePixelType, DestinationPixelType>::LearnStep0(const ImageReadable<PixelType, ReferencePixelType>& reference, QuadraticOptimization & opt)
	{
		for (int j = 2; j < src.Height() - 2; j++)
			for (int i = 2; i < src.Width() - 2; i++)
			{
				EDResampling::CoefficientsStep0<float> c = this->GetCoefficientsStep0(src.get_imgf(), i, j);
				opt.AddData(c.data, 6, (float)reference(i * 2, j * 2));
			}
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class ReferencePixelType>
	void EDResampling::Learner<PixelType, SourcePixelType, DestinationPixelType>::LearnStep1(const ImageReadable<PixelType, ReferencePixelType>& reference, QuadraticOptimization & opt)
	{
		weights.CalcStep1(src.get_imgf());

		for (int j = 1; j < src.Height() - 2; j++)
		{
			for (int i = 1; i < src.Width() - 2; i++)
			{
				EDResampling::CoefficientsStep1<float> c = this->GetCoefficientsStep1(src.get_imgf(), i, j);
				float w = weights.get(i * 2 + 1, j * 2 + 1);
				float nw = 1.0f - w;

				float v[6] = { c.data[0] * w + c.data[3] * nw,
					           c.data[1] * w + c.data[2] * nw,
					           c.data[2] * w + c.data[1] * nw,
					           c.data[3] * w + c.data[0] * nw,
					           c.data[4] * w + c.data[5] * nw,
					           c.data[5] * w + c.data[4] * nw };

				opt.AddData(v, 6, (float)reference(i * 2 + 1, j * 2 + 1));
			}
		}
	}

	template<class PixelType, class SourcePixelType, class DestinationPixelType> template<class ReferencePixelType>
	void EDResampling::Learner<PixelType, SourcePixelType, DestinationPixelType>::LearnStep2(const ImageReadable<PixelType, ReferencePixelType>& reference, QuadraticOptimization & opt2, QuadraticOptimization & opt2a, QuadraticOptimization & opt2b, QuadraticOptimization & opt2c)
	{
		weights.CalcStep2(dst.get_imgf());

		for (int j = 5; j < dst.Height() - 5; j++)
		{
			for (int i = 5 + (j % 2); i < src.Width() - 5; i += 2)
			{
				EDResampling::CoefficientsStep2<float> c = this->GetCoefficientsStep2(dst.get_imgf(), i, j);
				float w = weights.get(i, j);
				float nw = 1.0f - w;

				float v[6] = { c.data[0] * w + c.data[3] * nw,
					c.data[1] * w + c.data[2] * nw,
					c.data[2] * w + c.data[1] * nw,
					c.data[3] * w + c.data[0] * nw,
					c.data[4] * w + c.data[5] * nw,
					c.data[5] * w + c.data[4] * nw };

				opt2.AddData(v, 6, (float)reference(i, j));

				float v1[12] = { c.data[0] * w,  c.data[1] * w,  c.data[2] * w,  c.data[3] * w,  c.data[4] * w,  c.data[5] * w,
					             c.data[0] * nw, c.data[1] * nw, c.data[2] * nw, c.data[3] * nw, c.data[4] * nw, c.data[5] * nw };

				float v2[12] = { c.data[3] * nw, c.data[2] * nw, c.data[1] * nw, c.data[0] * nw, c.data[4] * nw, c.data[4] * nw,
             			    	 c.data[3] * w,  c.data[2] * w,  c.data[1] * w,  c.data[0] * w,  c.data[5] * w,  c.data[4] * w };

				if (j % 2 == 0)
				{
					opt2a.AddData(v1, 12, (float)reference(i, j));
					opt2c.AddData(v1, 12, (float)reference(i, j));
				}
				else
				{
					opt2b.AddData(v1, 12, (float)reference(i, j));
					opt2c.AddData(v2, 12, (float)reference(i, j));
				}
			}
		}
	}

	// =====================================================================================================================================

	template<class ImageType>
	inline auto EDResampling::SafeWrap(const ImageType & img) -> EDResampling::SafeWrapImpl<decltype(img(0, 0)), ImageType>
	{
		return EDResampling::SafeWrapImpl<decltype(img(0, 0)), ImageType>(img);
	}

	template <class PixelType, class SourceImageType, class DestinationImageType>
	void EDResampling::Perform(const ImageReadable<PixelType, SourceImageType> & src, ImageWritable<PixelType, DestinationImageType> & dst, EDResampling::Mode mode)
	{
		EDResampling::Resampler<PixelType, SourceImageType, DestinationImageType> resampler(src, dst, kernels, fast_weights);
		resampler.Perform(mode);
	}

	// =====================================================================================================================================

	template<class PixelType, class SourceImageType, class DestinationImageType>
	void EDResamplingExt::LearnStep1(const ImageReadable<PixelType, SourceImageType>& lr, const ImageReadable<PixelType, DestinationImageType>& hr)
	{
		Image<PixelType> tmp(lr.Width() * 2, lr.Height() * 2);

		EDResampling::Learner<PixelType, SourceImageType, BitmapImage<PixelType, true>> learner(lr, tmp, kernels, fast_weights);
		learner.LearnStep0(hr, opt0);
		learner.LearnStep1(hr, opt1);
	}

	template<class PixelType, class SourceImageType, class DestinationImageType>
	void EDResamplingExt::LearnStep2(const ImageReadable<PixelType, SourceImageType>& lr, const ImageReadable<PixelType, DestinationImageType>& hr)
	{
		Image<PixelType> tmp(lr.Width() * 2, lr.Height() * 2);

		EDResampling::Learner<PixelType, SourceImageType, BitmapImage<PixelType, true>> learner(lr, tmp, kernels, fast_weights);
		learner.InterpolateStep0();
		learner.InterpolateStep1();
		learner.LearnStep2(hr, opt2, opt2a, opt2b, opt2c);
	}

	// =====================================================================================================================================
}

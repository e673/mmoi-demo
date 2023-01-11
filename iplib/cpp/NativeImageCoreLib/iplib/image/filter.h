#pragma once

#include "core.h"
#include "../math/gauss_function.h"
#include <algorithm>
#include <utility>

namespace ip
{
	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType, class DestinationImageType, typename FilterPixelType>
	void FilterHorizontal(const ImageReadable<SourcePixelType, SourceImageType> &src, ImageWritable<DestinationPixelType, DestinationImageType> &dst,
		FilterPixelType *filter, int filter_len, int filter_center)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());
		check(filter_len <= src.Width());

		typedef decltype(std::declval<SourcePixelType>() * std::declval<float>()) IntermediateType;

		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = filter_center; i < filter_len; i++)
			{
				IntermediateType s = IntermediateType();
				for (int k = 0; k < filter_len /* i */; k++)
					s += src((std::max)(i - k, 0), j) * filter[k];
				dst(i - filter_center, j) = static_cast<DestinationPixelType>(s);
			}

			for (int i = filter_len; i < src.Width(); i++)
			{
				IntermediateType s = IntermediateType();
				for (int k = 0; k < filter_len; k++)
					s += src(i - k, j) * filter[k];
				dst(i - filter_center, j) = static_cast<DestinationPixelType>(s);
			}

			for (int i = 0; i < filter_center; i++)
			{
				IntermediateType s = IntermediateType();
				for (int k = 0 /* i + 1 */; k < filter_len; k++)
					s += src(src.Width() + (std::min)(i - k, -1), j) * filter[k];
				dst(src.Width() + i - filter_center, j) = static_cast<DestinationPixelType>(s);
			}
		}
	}

	template <class SourcePixelType, class SourceImageType>
	class InvertReadable
		: public ImageReadable<SourcePixelType, InvertReadable<SourcePixelType, SourceImageType>>
	{
	private:
		ImageReadable<SourcePixelType, SourceImageType> &src;

	public:
		InvertReadable(ImageReadable<SourcePixelType, SourceImageType> &src)
			: src(src) {}

		int Width() const
		{
			return src.Height();
		}

		int Height() const
		{
			return src.Width();
		}

		SourcePixelType pixel(int x, int y) const
		{
			return src.pixel(y, x);
		}
	};

	template <class SourcePixelType, class SourceImageType>
	class InvertWritable
		: public ImageWritable<SourcePixelType, InvertWritable<SourcePixelType, SourceImageType>>
	{
	private:
		ImageWritable<SourcePixelType, SourceImageType> &src;

	public:
		InvertWritable(ImageWritable<SourcePixelType, SourceImageType> &src)
			: src(src) {}

		int Width() const
		{
			return src.Height();
		}

		int Height() const
		{
			return src.Width();
		}

		SourcePixelType& pixel(int x, int y)
		{
			return src.pixel(y, x);
		}
	};

	template <class SourcePixelType, class SourceImageType>
	InvertReadable<SourcePixelType, SourceImageType> Invert(const ImageReadable<SourcePixelType, SourceImageType> &src)
	{
		return InvertReadable<SourcePixelType, SourceImageType>(src);
	}

	template <class SourcePixelType, class SourceImageType>
	InvertWritable<SourcePixelType, SourceImageType> Invert(ImageWritable<SourcePixelType, SourceImageType> &src)
	{
		return InvertWritable<SourcePixelType, SourceImageType>(src);
	}

	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType, class DestinationImageType>
	void DerivativeX(const ImageReadable<SourcePixelType, SourceImageType> &src, ImageWritable<DestinationPixelType, DestinationImageType> &dst, float sigma)
	{
		GaussFunction gf(sigma);
		GaussFunctionDerivative gf2(sigma);

		int rad = (int)(sigma * 3.0f);

		float *v1 = new float[2 * rad + 1], *v2 = new float[2 * rad + 1];
		for (int i = -rad; i <= rad; i++)
		{
			v1[i + rad] = gf((float)i);
			v2[i + rad] = gf2((float)i);
		}

		typedef decltype(src(0, 0) * v1[0]) IntermediateType;

		Image<IntermediateType> tmp(src.Height(), src.Width());

		FilterHorizontal(src, Invert(tmp), v2, 2 * rad + 1, rad);
		FilterHorizontal(tmp, Invert(dst), v1, 2 * rad + 1, rad);

		delete[] v1;
		delete[] v2;
	}

	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType, class DestinationImageType>
	void DerivativeY(const ImageReadable<SourcePixelType, SourceImageType> &src, ImageWritable<DestinationPixelType, DestinationImageType> &dst, float sigma)
	{
		GaussFunction gf(sigma);
		GaussFunctionDerivative gf2(sigma);

		int rad = (int)(sigma * 3.0f);

		float *v1 = new float[2 * rad + 1], *v2 = new float[2 * rad + 1];
		for (int i = -rad; i <= rad; i++)
		{
			v1[i + rad] = gf(i);
			v2[i + rad] = gf2(i);
		}

		typedef decltype(src(0, 0) * v1[0]) IntermediateType;

		Image<IntermediateType> tmp(src.Height(), src.Width());
		FilterHorizontal(src, Invert(tmp), v1, 2 * rad + 1, rad);
		FilterHorizontal(tmp, Invert(dst), v2, 2 * rad + 1, rad);

		delete[] v1;
		delete[] v2;
	}

	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType, class DestinationImageType>
	void GaussFilter(const ImageReadable<SourcePixelType, SourceImageType> &src, ImageWritable<DestinationPixelType, DestinationImageType> &dst, float sigma)
	{
		GaussFunction gf(sigma);

		int rad = (int)(sigma * 3.0f);

		float *v1 = new float[2 * rad + 1];
		for (int i = -rad; i <= rad; i++)
			v1[i + rad] = gf((float)i);

		typedef decltype(src(0, 0) * v1[0]) IntermediateType;

		Image<IntermediateType> tmp(src.Height(), src.Width());
		FilterHorizontal(src, Invert(tmp), v1, 2 * rad + 1, rad);
		FilterHorizontal(tmp, Invert(dst), v1, 2 * rad + 1, rad);

		delete[] v1;
	}

	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType>
	class ImageCast
		: public ImageReadable<DestinationPixelType, ImageCast<SourcePixelType, SourceImageType, DestinationPixelType>>
	{
		const ImageReadable<SourcePixelType, SourceImageType> &img;

	public:
		ImageCast(const ImageReadable<SourcePixelType, SourceImageType> &img)
			: img(img) {}

		int Width() const { return img.Width(); }
		int Height() const { return img.Height(); }
		DestinationPixelType pixel(int x, int y) { return (DestinationPixelType)img.pixel(x, y); }
	};

	template <typename DestinationPixelType, typename SourcePixelType, class SourceImageType>
	ImageCast<SourcePixelType, SourceImageType, DestinationPixelType> Cast(const ImageReadable<SourcePixelType, SourceImageType> &img)
	{
		return ImageCast<SourcePixelType, SourceImageType, DestinationPixelType>(img);
	}

	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType, class DestinationImageType, typename FilterPixelType, class FilterImageType>
	void Filter2D(const ImageReadable<SourcePixelType, SourceImageType> &src, ImageWritable<DestinationPixelType, DestinationImageType> &dst,
		const ImageReadable<FilterPixelType, FilterImageType> &filter, int filter_center_x, int filter_center_y)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());

		for (int j = 0; j < src.Height(); j++)
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
		}
	}
}
#pragma once

#include "core.h"

namespace ip
{
	template <typename SourcePixelType, class SourceImageType, typename DestinationPixelType, class TransformOperation>
	class PixelTransformImage
		: public ImageReadable<DestinationPixelType, PixelTransformImage<SourcePixelType, SourceImageType, DestinationPixelType, TransformOperation>>
	{
	private:
		const ImageReadable<SourcePixelType, SourceImageType> &src;
		TransformOperation operation;

	public:
		PixelTransformImage(const ImageReadable<SourcePixelType, SourceImageType> &src, const TransformOperation &operation)
			: src(src), operation(operation) {}

		int Width() const
		{
			return src.Width();
		}

		int Height() const
		{
			return src.Height();
		}

		DestinationPixelType pixel(int x, int y) const
		{
			return operation(src(x, y));
		}
	};

	template <typename SourcePixelType1, class SourceImageType1, typename SourcePixelType2, class SourceImageType2, typename DestinationPixelType, class TransformOperation>
	class PixelTransformImage2
		: public ImageReadable<DestinationPixelType, PixelTransformImage2<SourcePixelType1, SourceImageType1, SourcePixelType2, SourceImageType2, DestinationPixelType, TransformOperation>>
	{
	private:
		const ImageReadable<SourcePixelType1, SourceImageType1> &src1;
		const ImageReadable<SourcePixelType2, SourceImageType2> &src2;
		TransformOperation operation;

	public:
		PixelTransformImage2(const ImageReadable<SourcePixelType1, SourceImageType1> &src1, const ImageReadable<SourcePixelType2, SourceImageType2> &src2, const TransformOperation &operation)
			: src1(src1), src2(src2), operation(operation)
		{
			check(src1.Width() == src2.Width() && src1.Height() == src2.Height());
		}

		int Width() const
		{
			return src1.Width();
		}

		int Height() const
		{
			return src1.Height();
		}

		DestinationPixelType pixel(int x, int y) const
		{
			return operation(src1(x, y), src2(x, y));
		}
	};

	template <typename SourcePixelType, class SourceImageType, class TransformOperation>
	auto PixelTransform(const ImageReadable<SourcePixelType, SourceImageType> &src, const TransformOperation &operation)
		-> PixelTransformImage<SourcePixelType, SourceImageType, typename decltype(operation(std::declval<SourcePixelType>())), TransformOperation>
	{
		return PixelTransformImage<SourcePixelType, SourceImageType, typename decltype(operation(std::declval<SourcePixelType>())), TransformOperation>(src, operation);
	}

	template <typename SourcePixelType1, class SourceImageType1, typename SourcePixelType2, class SourceImageType2, class TransformOperation>
	auto PixelTransform(const ImageReadable<SourcePixelType1, SourceImageType1> &src1, const ImageReadable<SourcePixelType2, SourceImageType2> &src2, const TransformOperation &operation)
		-> PixelTransformImage2<SourcePixelType1, SourceImageType1, SourcePixelType2, SourceImageType2,
		typename decltype(operation(std::declval<SourcePixelType1>(), std::declval<SourcePixelType2>())), TransformOperation>
	{
		return PixelTransformImage2<SourcePixelType1, SourceImageType1, SourcePixelType2, SourceImageType2,
			typename decltype(operation(std::declval<SourcePixelType1>(), std::declval<SourcePixelType2>())), TransformOperation>(src1, src2, operation);
	}

}
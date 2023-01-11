#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "../base/imagereadable.h"

namespace ip
{
	// ==================================================================================================
	//                                   ImageUnaryOperation - interface               
	// ==================================================================================================

	template <typename PixelType, class SourceImageType, class UnaryOperation>
	class ImageUnaryOperation
		: public ImageReadable<typename UnaryOperation::PixelType, ImageUnaryOperation<PixelType, SourceImageType, UnaryOperation>>
	{
	public:
		const ImageBase<PixelType, SourceImageType> &src;
		const UnaryOperation operation;

		ImageUnaryOperation(const ImageBase<PixelType, SourceImageType> &src, const UnaryOperation &operation);
		int Width() const;
		int Height() const;
		typename UnaryOperation::PixelType pixel(int x, int y) const;
	};

	// ==================================================================================================
	//                                  ImageUnaryOperation - implementation               
	// ==================================================================================================

	template <typename PixelType, class SourceImageType, class UnaryOperation>
	ImageUnaryOperation<PixelType, SourceImageType, UnaryOperation>::ImageUnaryOperation(const ImageBase<PixelType, SourceImageType> &src, const UnaryOperation &operation)
		: src(src), operation(operation) {}

	template <typename PixelType, class SourceImageType, class UnaryOperation>
	int ImageUnaryOperation<PixelType, SourceImageType, UnaryOperation>::Width() const
	{
		return src.Width();
	}

	template <typename PixelType, class SourceImageType, class UnaryOperation>
	int ImageUnaryOperation<PixelType, SourceImageType, UnaryOperation>::Height() const
	{
		return src.Height();
	}

	template <typename PixelType, class SourceImageType, class UnaryOperation>
	typename UnaryOperation::PixelType ImageUnaryOperation<PixelType, SourceImageType, UnaryOperation>::pixel(int x, int y) const
	{
		return operation(src.pixel(x, y));
	}
}
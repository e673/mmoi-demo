#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "../base/imagereadable.h"

namespace ip
{
	// ==================================================================================================
	//                                   ImageBinaryOperation - interface               
	// ==================================================================================================

	template <class SourceImageType1, class SourceImageType2, class BinaryOperation>
	class ImageBinaryOperation
		: public ImageReadable<typename BinaryOperation::PixelType, ImageBinaryOperation<SourceImageType1, SourceImageType2, BinaryOperation>>
	{
	public:
		const SourceImageType1 &src1;
		const SourceImageType2 &src2;
		const BinaryOperation operation;

		ImageBinaryOperation(const SourceImageType1 &src1, const SourceImageType2 &src2, const BinaryOperation &operation);
		int Width() const;
		int Height() const;
		typename BinaryOperation::PixelType pixel(int x, int y) const;
	};

	// ==================================================================================================
	//                                  ImageBinaryOperation - implementation               
	// ==================================================================================================

	template <class SourceImageType1, class SourceImageType2, class BinaryOperation>
	ImageBinaryOperation<SourceImageType1, SourceImageType2, BinaryOperation>::ImageBinaryOperation(const SourceImageType1 &src1, const SourceImageType2 &src2, const BinaryOperation &operation)
		: src1(src1), src2(src2), operation(operation) {}

	template <class SourceImageType1, class SourceImageType2, class BinaryOperation>
	int ImageBinaryOperation<SourceImageType1, SourceImageType2, BinaryOperation>::Width() const
	{
		return src1.Width();
	}

	template <class SourceImageType1, class SourceImageType2, class BinaryOperation>
	int ImageBinaryOperation<SourceImageType1, SourceImageType2, BinaryOperation>::Height() const
	{
		return src1.Height();
	}

	template <class SourceImageType1, class SourceImageType2, class BinaryOperation>
	typename BinaryOperation::PixelType ImageBinaryOperation<SourceImageType1, SourceImageType2, BinaryOperation>::pixel(int x, int y) const
	{
		return operation(src1.pixel(x, y), src2.pixel(x, y));
	}
}
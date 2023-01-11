#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebase.h"

namespace ip
{
	// ==================================================================================================
	//                                   Interface               
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	class ImageReadable
		: public ImageBase < PixelType , ImageType >
	{
		// This class can be specialized for different pixel/image types

	public:
		template <typename IntermediatePixelType, typename ResultPixelType, class ResultImageType>
		void FilterGauss(ImageWritable<ResultPixelType, ResultImageType> &dst, float sigma, int wnd_size = 0);
	};
}
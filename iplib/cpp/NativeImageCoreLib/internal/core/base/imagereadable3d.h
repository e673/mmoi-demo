#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebase3d.h"

namespace ip
{
	// ==================================================================================================
	//                                   Interface               
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	class ImageReadable3D
		: public ImageBase3D < PixelType, ImageType >
	{
		// This class can be specialized for different pixel/image types
	};
}
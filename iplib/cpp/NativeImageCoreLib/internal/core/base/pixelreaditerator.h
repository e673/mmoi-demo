#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebasefwd.h"

namespace ip
{
	// ==================================================================================================
	//                                   PixelReadIterator - interface              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	class PixelReadIterator
	{
	public:
		// Create default pixel read iterator
		PixelReadIterator(const ImageBase<PixelType, ImageType>& image, int x, int y);

		// Get the pixel value with offset N from the current position
		PixelType get(int PixelOffset = 0);

		// Advance the current position by N pixels
		void advance(int PixelOffset);

	private:
		const ImageBase<PixelType, ImageType>& image;
		int x, y;
	};

	// ==================================================================================================
	//                                   PixelReadIterator - implementation              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	PixelReadIterator<PixelType, ImageType, Enable>::PixelReadIterator(const ImageBase<PixelType, ImageType>& image, int x, int y)
		: image(image), x(x), y(y) {}

	template <typename PixelType, class ImageType, typename Enable>
	PixelType PixelReadIterator<PixelType, ImageType, Enable>::get(int PixelOffset)
	{
		return image.pixel(x + PixelOffset, y);
	}

	template <typename PixelType, class ImageType, typename Enable>
	void PixelReadIterator<PixelType, ImageType, Enable>::advance(int PixelOffset)
	{
		x += PixelOffset;
	}
}
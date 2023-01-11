#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebasefwd.h"

#include <immintrin.h>

namespace ip
{
	// ==================================================================================================
	//                                   VectorReadIterator - interface              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	class VectorReadIterator
	{
	public:
		// Create default pixel read iterator
		VectorReadIterator(const ImageBase<PixelType, ImageType>& image, int x, int y);

		/* Get the 256-bit pixel value with offset BlockOffset (in 256-bit words) from the current position.
		* The pixel cycle size is always 32 pixels, so the block cycle size equals to pixel size
		* The parameter RemaniningPixels sets the number of pixels in the block for the last block if its size is less than 32 pixels */
		template <int BlockOffset, int RemainingPixels = 32>
		__m256 get256();

		// Advance the current position by N pixels
		void advance(int PixelOffset);

	private:
		PixelReadIterator<PixelType, ImageType> reader;
		PixelType buf[32];

		template <int RemainingPixels>
		struct helper
		{
			template <int BlockOffset>
			static __m256 get(VectorReadIterator<PixelType, ImageType>& vr)
			{
				return _mm256_loadu_ps((float*)vr.buf + BlockOffset * 8);
			}

			template <>
			static __m256 get<0>(VectorReadIterator<PixelType, ImageType>& vr)
			{
				for (int i = 0; i < RemainingPixels; i++)
					vr.buf[i] = vr.reader.get(i);
				return _mm256_loadu_ps((float*)vr.buf);
			}
		};
	};

	// ==================================================================================================
	//                                   VectorReadIterator - implementation              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	VectorReadIterator<PixelType, ImageType, Enable>::VectorReadIterator(const ImageBase<PixelType, ImageType>& image, int x, int y)
		: reader(image, x, y) {}

	template <typename PixelType, class ImageType, typename Enable>
	template <int BlockOffset, int RemainingPixels>
	__m256 VectorReadIterator<PixelType, ImageType, Enable>::get256()
	{
		return helper<RemainingPixels>::get<BlockOffset>(*this);
	}

	template <typename PixelType, class ImageType, typename Enable>
	void VectorReadIterator<PixelType, ImageType, Enable>::advance(int PixelOffset)
	{
		reader.advance(PixelOffset);
	}
}
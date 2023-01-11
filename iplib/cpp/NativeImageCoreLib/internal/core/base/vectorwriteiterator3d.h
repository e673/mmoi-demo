#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebasefwd.h"

namespace ip
{
	// ==================================================================================================
	//                                   VectorWriteIterator - interface              
	// ==================================================================================================

	/* Default implementation of pixel write iterator */
	template <typename PixelType, class ImageType, typename Enable>
	class VectorWriteIterator3D
	{
	public:

		// Create default pixel write iterator
		VectorWriteIterator3D(ImageWritable3D<PixelType, ImageType>& image, int x, int y, int z);

		/* Set the 256-bit pixel value with offset BlockOffset (in 256-bit words) from the current position.
		* The pixel cycle size is always 32 pixels, so the block cycle size equals to pixel size
		* The parameter RemaniningPixels sets the number of pixels in the block for the last block if its size is less than 32 pixels */
		template <int BlockOffset, int RemainingPixels = 32>
		void set256(__m256 value);

		// Advance the current position by N pixels
		void advance(int N);

	private:
		PixelWriteIterator3D<PixelType, ImageType> writer;
		PixelType buf[32];

		template <int RemainingPixels>
		struct helper
		{
			static const int LastBlock = (RemainingPixels * sizeof(PixelType) - 1) / 32;

			template <int BlockOffset>
			static void set(VectorReadIterator<PixelType, ImageType>& vr, __m256 value)
			{
				_mm256_storeu_ps((float*)vr.buf + BlockOffset * 8, value);
			}

			template <>
			static void set<LastBlock>(VectorReadIterator<PixelType, ImageType>& vr, __m256 value)
			{
				for (int i = 0; i < RemainingPixels; i++)
					vr.writer.set(i, vr.buf[i]);
			}
		};
	};

	// ==================================================================================================
	//                                   VectorWriteIterator - implementation              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	VectorWriteIterator3D<PixelType, ImageType, Enable>::VectorWriteIterator3D(ImageWritable3D<PixelType, ImageType>& image, int x, int y, int z)
		: writer(image, x, y, z) {}

	template <typename PixelType, class ImageType, typename Enable>
	template <int BlockOffset, int RemainingPixels>
	void VectorWriteIterator3D<PixelType, ImageType, Enable>::set256(__m256 value)
	{
		helper<RemainingPixels>::set<BlockOffset>(*this, value);
	}

	template <typename PixelType, class ImageType, typename Enable>
	void VectorWriteIterator3D<PixelType, ImageType, Enable>::advance(int PixelOffset)
	{
		writer.advance(PixelOffset);
	}
}
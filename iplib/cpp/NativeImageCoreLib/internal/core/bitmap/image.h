#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "bitmapimage.h"
#include "../base/pixeltypes.h"

namespace ip
{
	namespace internal
	{
		class BitmapData
			: public BitmapDataResource<BitmapDataStructure>
		{
		public:
			BitmapData(int Width, int Height, int PixelSize);
			virtual ~BitmapData();

			static const int Alignment = 32;
		};
	}

	// ==========================================================================================

	template <typename PixelType>
	class Image
		: public BitmapImage<PixelType, true>
	{
	public:
		Image()
		{
			this->data.data = nullptr;
			this->data.width = 0;
			this->data.height = 0;
			this->data.stride = 0;
		}

		Image(int Width, int Height)
		{
			this->base_data.reset(new internal::BitmapData(Width, Height, sizeof(PixelType)));
			this->data = *this->base_data;
		}

		Image(Image<PixelType> &&move_from)
			: Image()
		{
			swap(move_from);
		}

		template <class ImageType>
		Image(const ImageReadable<PixelType, ImageType> &copy_from)
			: Image(copy_from.Width(), copy_from.Height())
		{
			for (int j = 0; j < this->Height(); j++)
			{
				for (int i = 0; i < this->Width(); i++)
				{
					this->pixel(i, j) = copy_from.pixel(i, j);
				}
			}
		}

		void SetSize(int newWidth, int newHeight)
		{
			check(this->base_data);
			check(newWidth > 0 && newWidth <= this->base_data->width);
			check(newHeight > 0 && newHeight <= this->base_data->height);

			this->data.width = newWidth;
			this->data.height = newHeight;
		}

		// --------------------------------------------------------------------------------------

		void swap(Image<PixelType> &other)
		{
			this->data = other.data;
			this->base_data.swap(other.base_data);
			other.data.data = nullptr;
			other.data.width = other.data.height = 0;
			other.data.stride = 0;
		}
	};

	// ==========================================================================================


	typedef Image<void> UntypedImage;
	typedef Image<bool> ImageBinary;
	typedef Image<byte> ImageByte;
	typedef Image<PixelByteRGB> ImageByteColor3;
	typedef Image<PixelByteRGB> ImageByteRGB;
	typedef Image<PixelByteRGB4> ImageByteColor4;
	typedef Image<PixelByteRGB4> ImageByteRGB4;
	typedef Image<PixelByteRGBA> ImageByteColor;
	typedef Image<PixelByteRGBA> ImageByteRGBA;
	typedef Image<float> ImageFloat;
	typedef Image<PixelFloatRGB> ImageFloatColor3;
	typedef Image<PixelFloatRGB> ImageFloatRGB;
	typedef Image<PixelFloatRGB4> ImageFloatColor4;
	typedef Image<PixelFloatRGB4> ImageFloatRGB4;
	typedef Image<PixelFloatRGBA> ImageFloatColor;
	typedef Image<PixelFloatRGBA> ImageFloatRGBA;
	typedef Image<PixelFloatComplex> ImageComplex;

	typedef Image<PixelFloatVector> ImageVectorFloat;
}
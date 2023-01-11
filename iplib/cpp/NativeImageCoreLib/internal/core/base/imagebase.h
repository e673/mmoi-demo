#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebasefwd.h"

#include "../ops/imageunaryoperation.h"
#include "../ops/convert.h"

namespace ip
{
	// ==================================================================================================
	//                                   ImageBase : interface               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	class ImageBase
	{
	public:
		typedef PixelReadIterator<PixelType, ImageType> PixelReader;
		typedef VectorReadIterator<PixelType, ImageType> VectorReader;

	public:
		int Width() const;
		int Height() const;

		PixelType pixel(int x, int y) const;
		PixelType operator() (int x, int y) const;

		PixelReader GetPixelReadIterator(int x, int y) const;
		VectorReader GetVectorReadIterator(int x, int y) const;

		// ----------------------------------------------------------------------------------------------

		template <typename DestinationPixelType>
		ImageUnaryOperation<PixelType, ImageType, OperationConvert<DestinationPixelType, PixelType>> Convert() const
		{
			return ImageUnaryOperation<PixelType, ImageType, OperationConvert<DestinationPixelType, PixelType>>(*this, OperationConvert<DestinationPixelType, PixelType>());

			// ToDo
		}


		template <class OtherImageType>
		void CopyTo(ImageWritable<PixelType, OtherImageType> &dst) const;
	};

	// ==================================================================================================
	//                                   ImageBase : static polymorphism
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	int ImageBase<PixelType, ImageType>::Width() const
	{
		return static_cast<const ImageType*>(this)->Width();
	}

	template <typename PixelType, class ImageType>
	int ImageBase<PixelType, ImageType>::Height() const
	{
		return static_cast<const ImageType*>(this)->Height();
	}

	template <typename PixelType, class ImageType>
	PixelType ImageBase<PixelType, ImageType>::pixel(int x, int y) const
	{
		return static_cast<const ImageType*>(this)->pixel(x, y);
	}

	template <typename PixelType, class ImageType>
	PixelType ImageBase<PixelType, ImageType>::operator() (int x, int y) const
	{
		return static_cast<const ImageType*>(this)->pixel(x, y);
	}

	// ==================================================================================================
	//                              ImageBase : general image processing operations               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	PixelReadIterator<PixelType, ImageType> ImageBase<PixelType, ImageType>::GetPixelReadIterator(int x, int y) const
	{
		return PixelReadIterator<PixelType, ImageType>(static_cast<const ImageType&>(*this), x, y);

	}

	template <typename PixelType, class ImageType>
	VectorReadIterator<PixelType, ImageType> ImageBase<PixelType, ImageType>::GetVectorReadIterator(int x, int y) const
	{
		return VectorReadIterator<PixelType, ImageType>(static_cast<const ImageType&>(*this), x, y);
	}

	template <typename PixelType, class ImageType>
	template <class OtherImageType>
	void ImageBase<PixelType, ImageType>::CopyTo(ImageWritable<PixelType, OtherImageType> &dst) const
	{
		for (int j = 0; j < Height(); j++)
		{
			PixelReader reader = GetPixelReadIterator(0, j);
			typename ImageWritable<PixelType, OtherImageType>::PixelWriter writer = dst.GetPixelWriteIterator(0, j);

			for (int i = 0; i < Width(); i++)
			{
				writer.set(reader.get());
				reader.advance(1);
				writer.advance(1);
			}
		}
	}
}
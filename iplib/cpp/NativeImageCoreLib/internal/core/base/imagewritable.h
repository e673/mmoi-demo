#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagereadable.h"

namespace ip
{
	// ==================================================================================================
	//                                   ImageWritable - interface               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	class ImageWritable
		: public ImageReadable<PixelType, ImageType>
	{
	public:
		typedef PixelWriteIterator<PixelType, ImageType> PixelWriter;
		typedef VectorWriteIterator<PixelType, ImageType> VectorWriter;

	public:
		PixelWriter GetPixelWriteIterator(int x, int y);
		VectorWriter GetVectorWriteIterator(int x, int y);

		PixelType& pixel(int x, int y);
		PixelType& operator()(int x, int y);

		PixelType pixel(int x, int y) const;
		PixelType operator()(int x, int y) const;
	};

	// ==================================================================================================
	//                                   ImageWritable - implementation               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	PixelWriteIterator<PixelType, ImageType> ImageWritable<PixelType, ImageType>::GetPixelWriteIterator(int x, int y)
	{
		return PixelWriteIterator<PixelType, ImageType>(static_cast<ImageType&>(*this), x, y);

	}

	template <typename PixelType, class ImageType>
	VectorWriteIterator<PixelType, ImageType> ImageWritable<PixelType, ImageType>::GetVectorWriteIterator(int x, int y)
	{
		return VectorWriteIterator<PixelType, ImageType>(static_cast<ImageType&>(*this), x, y);
	}

	template <typename PixelType, class ImageType>
	PixelType& ImageWritable<PixelType, ImageType>::pixel(int x, int y)
	{
		return (static_cast<ImageType*>(this))->pixel(x, y);
	}

	template <typename PixelType, class ImageType>
	PixelType& ImageWritable<PixelType, ImageType>::operator()(int x, int y)
	{
		return (static_cast<ImageType*>(this))->pixel(x, y);
	}


	template <typename PixelType, class ImageType>
	PixelType ImageWritable<PixelType, ImageType>::pixel(int x, int y) const
	{
		return (static_cast<const ImageType*>(this))->pixel(x, y);
	}

	template <typename PixelType, class ImageType>
	PixelType ImageWritable<PixelType, ImageType>::operator()(int x, int y) const
	{
		return (static_cast<const ImageType*>(this))->pixel(x, y);
	}
}
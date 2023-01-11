#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagereadable3d.h"

namespace ip
{
	// ==================================================================================================
	//                                   ImageWritable3D - interface               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	class ImageWritable3D
		: public ImageReadable3D<PixelType, ImageType>
	{
	public:
		typedef PixelWriteIterator3D<PixelType, ImageType> PixelWriter;
		typedef VectorWriteIterator3D<PixelType, ImageType> VectorWriter;

	public:
		PixelWriter GetPixelWriteIterator(int x, int y, int z);
		VectorWriter GetVectorWriteIterator(int x, int y, int z);

		PixelType& pixel(int x, int y, int z);
		PixelType& operator()(int x, int y, int z);

		PixelType pixel(int x, int y, int z) const;
		PixelType operator()(int x, int y, int z) const;
	};

	// ==================================================================================================
	//                                   ImageWritable3D - implementation               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	PixelWriteIterator3D<PixelType, ImageType> ImageWritable3D<PixelType, ImageType>::GetPixelWriteIterator(int x, int y, int z)
	{
		return PixelWriteIterator3D<PixelType, ImageType>(static_cast<ImageType&>(*this), x, y, z);

	}

	template <typename PixelType, class ImageType>
	VectorWriteIterator3D<PixelType, ImageType> ImageWritable3D<PixelType, ImageType>::GetVectorWriteIterator(int x, int y, int z)
	{
		return VectorWriteIterator3D<PixelType, ImageType>(static_cast<ImageType&>(*this), x, y, z);
	}

	template <typename PixelType, class ImageType>
	PixelType& ImageWritable3D<PixelType, ImageType>::pixel(int x, int y, int z)
	{
		return (static_cast<ImageType*>(this))->pixel(x, y, z);
	}

	template <typename PixelType, class ImageType>
	PixelType& ImageWritable3D<PixelType, ImageType>::operator()(int x, int y, int z)
	{
		return (static_cast<ImageType*>(this))->pixel(x, y, z);
	}

	template <typename PixelType, class ImageType>
	PixelType ImageWritable3D<PixelType, ImageType>::pixel(int x, int y, int z) const
	{
		return (static_cast<const ImageType*>(this))->pixel(x, y, z);
	}

	template <typename PixelType, class ImageType>
	PixelType ImageWritable3D<PixelType, ImageType>::operator()(int x, int y, int z) const
	{
		return (static_cast<const ImageType*>(this))->pixel(x, y, z);
	}
}
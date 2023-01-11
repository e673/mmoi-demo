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
	//                                   ImageBase3D : interface               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	class ImageBase3D
	{
	public:
		typedef PixelReadIterator3D<PixelType, ImageType> PixelReader;
		typedef VectorReadIterator3D<PixelType, ImageType> VectorReader;

	public:
		int SizeX() const;
		int SizeY() const;
		int SizeZ() const;

		PixelType pixel(int x, int y, int z) const;
		PixelType operator() (int x, int y, int z) const;

		PixelReader GetPixelReadIterator(int x, int y, int z) const;
		VectorReader GetVectorReadIterator(int x, int y, int z) const;

		// ----------------------------------------------------------------------------------------------

		template <class OtherImageType>
		void CopyTo(ImageWritable3D<PixelType, OtherImageType> &dst) const;
	};

	// ==================================================================================================
	//                                   ImageBase3D : static polymorphism
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	int ImageBase3D<PixelType, ImageType>::SizeX() const
	{
		return static_cast<const ImageType*>(this)->SizeX();
	}

	template <typename PixelType, class ImageType>
	int ImageBase3D<PixelType, ImageType>::SizeY() const
	{
		return static_cast<const ImageType*>(this)->SizeY();
	}

	template <typename PixelType, class ImageType>
	int ImageBase3D<PixelType, ImageType>::SizeZ() const
	{
		return static_cast<const ImageType*>(this)->SizeZ();
	}

	template <typename PixelType, class ImageType>
	PixelType ImageBase3D<PixelType, ImageType>::pixel(int x, int y, int z) const
	{
		return static_cast<const ImageType*>(this)->pixel(x, y, z);
	}

	template <typename PixelType, class ImageType>
	PixelType ImageBase3D<PixelType, ImageType>::operator() (int x, int y, int z) const
	{
		return static_cast<const ImageType*>(this)->pixel(x, y, z);
	}

	// ==================================================================================================
	//                              ImageBase3D : general image processing operations               
	// ==================================================================================================

	template <typename PixelType, class ImageType>
	PixelReadIterator3D<PixelType, ImageType> ImageBase3D<PixelType, ImageType>::GetPixelReadIterator(int x, int y, int z) const
	{
		return PixelReadIterator3D<PixelType, ImageType>(static_cast<const ImageType&>(*this), x, y, z);

	}

	template <typename PixelType, class ImageType>
	VectorReadIterator3D<PixelType, ImageType> ImageBase3D<PixelType, ImageType>::GetVectorReadIterator(int x, int y, int z) const
	{
		return VectorReadIterator3D<PixelType, ImageType>(static_cast<const ImageType&>(*this), x, y, z);
	}

	template <typename PixelType, class ImageType>
	template <class OtherImageType>
	void ImageBase3D<PixelType, ImageType>::CopyTo(ImageWritable3D<PixelType, OtherImageType> &dst) const
	{
		for (int z = 0; z < SizeZ(); z++)
		{
			for (int y = 0; y < SizeY(); y++)
			{
				PixelReader reader = GetPixelReadIterator(0, y, z);
				typename ImageWritable3D<PixelType, OtherImageType>::PixelWriter writer = dst.GetPixelWriteIterator(0, y, z);

				for (int x = 0; x < SizeX(); x++)
				{
					writer.set(reader.get());
					reader.advance(1);
					writer.advance(1);
				}
			}
		}
	}	
}
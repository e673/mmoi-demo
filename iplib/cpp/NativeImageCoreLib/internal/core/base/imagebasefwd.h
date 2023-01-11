#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

namespace ip
{
	template <typename PixelType, class ImageType>
	class ImageBase;

	template <typename PixelType, class ImageType, typename Enable = void>
	class PixelReadIterator;

	template <typename PixelType, class ImageType, typename Enable = void>
	class PixelWriteIterator;

	template <typename PixelType, class ImageType, typename Enable = void>
	class VectorReadIterator;

	template <typename PixelType, class ImageType, typename Enable = void>
	class VectorWriteIterator;

	template <typename PixelType, class ImageType, typename Enable = void>
	class ImageReadable;

	template <typename PixelType, class ImageType>
	class ImageWritable;

	// ==================================================================================================

	template <typename PixelType, class ImageType>
	class ImageBase3D;

	template <typename PixelType, class ImageType, typename Enable = void>
	class PixelReadIterator3D;

	template <typename PixelType, class ImageType, typename Enable = void>
	class PixelWriteIterator3D;

	template <typename PixelType, class ImageType, typename Enable = void>
	class VectorReadIterator3D;

	template <typename PixelType, class ImageType, typename Enable = void>
	class VectorWriteIterator3D;

	template <typename PixelType, class ImageType, typename Enable = void>
	class ImageReadable3D;

	template <typename PixelType, class ImageType>
	class ImageWritable3D;

	// ==================================================================================================

	template <typename PixelType, class SourceImageType, class UnaryOperation>
	class ImageUnaryOperation;
}
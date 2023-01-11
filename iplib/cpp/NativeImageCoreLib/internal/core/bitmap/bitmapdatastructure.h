#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

namespace ip
{
	// This structure is passed over modules - it's layout must be specified
#pragma pack(push, 1)

	class BitmapDataStructure
	{
	public:
		void *data;
		ptrdiff_t stride;
		int width;
		int height;

		BitmapDataStructure Fragment(const BitmapDataStructure &img, int OffsetX, int OffsetY, int Width, int Height, int PixelSize);
	};

	class BitmapDataStructure3D
	{
	public:
		void *data;
		ptrdiff_t stride_y, stride_z;
		int size_x, size_y, size_z;

		BitmapDataStructure3D Fragment(const BitmapDataStructure3D &img, int OffsetX, int OffsetY, int OffsetZ,	int SizeX, int SizeY, int SizeZ, int PixelSize);
	};

#pragma pack(pop)

	namespace internal
	{
		template <class BitmapDataStructureType>
		class BitmapDataResource
			: public BitmapDataStructureType
		{
		public:
			virtual ~BitmapDataResource() {}
		};
	}
}
#pragma once

#include "../../iplib/image/core3d.h"
#include "transimagebase.h"

namespace ip
{
	class TransImage3D
		: public TransImageBase < BitmapDataStructure3D >
	{
	public:
		void CreateImage3D(int SizeX, int SizeY, int SizeZ, int PixelSize);
		void CreateFragment(TransImage3D &img, int OfsX, int OfsY, int OfsZ, int SizeX, int SizeY, int SizeZ, int PixelSize);
	};
}
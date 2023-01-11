#pragma once

#include "../../iplib/image/core.h"
#include "../../iplib/image/io/dibitmap.h"
#include "transimagebase.h"

namespace ip
{
	class TransImage
		: public TransImageBase < BitmapDataStructure >
	{
	public:
		void CreateDIBitmap(int Width, int Height, HDC *hdc);
		void CreateImage(int Width, int Height, int PixelSize);
		void CreateFragment(TransImage &img, int OfsX, int OfsY, int Width, int Height, int PixelSize);
	};
}
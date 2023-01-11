#pragma once

#include <Windows.h>
#include "../core.h"

namespace ip
{
	namespace internal
	{
		class DIBitmapData
			: public BitmapDataResource<BitmapDataStructure>
		{
		public:
			DIBitmapData(int Width, int Height);
			virtual ~DIBitmapData();

		public:
			HBITMAP hbitmap;
			HGDIOBJ hold;
			HDC hdc;
		};
	}

	// ==========================================================================================

	class DIBitmap
		: public BitmapImage<PixelB4, false>
	{
	public:
		DIBitmap(int Width, int Height);
	};
}
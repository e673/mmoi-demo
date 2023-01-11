#pragma once

#include <iplib/image/core.h>

namespace ip
{
	namespace padding
	{
		enum class PaddingAccessMode
		{
			Left,
			Normal,
			Right
		};

		template <typename PixelType, class ImageType, PaddingAccessMode Horz, PaddingAccessMode Vert>
		class DuplicatePaddingAccessor
		{
			const ImageReadable<PixelType, ImageType> &src;

		public:
			DuplicatePaddingAccessor(const ImageReadable<PixelType, ImageType> &src)
				: src(src) {}

			PixelType operator () (int x, int y) const
			{
				return src(Horz == PaddingAccessMode::Normal ? x : (Horz == PaddingAccessMode::Left ? 0 : src.Width() - 1),
					Vert == PaddingAccessMode::Normal ? y : (Vert == PaddingAccessMode::Left ? 0 : src.Height() - 1));
			}
		};
	}
}
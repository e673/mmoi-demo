#pragma once

#include "../edt/edt.h"

namespace ip
{
	template <class SourceImageType>
	class InvertImage
		: public ImageReadable<bool, InvertImage<SourceImageType>>
	{
	private:
		const ImageReadable<bool, SourceImageType> &src;

	public:
		InvertImage(const ImageReadable<bool, SourceImageType> &src)
			: src(src) {}

		int Width() const
		{
			return src.Width();
		}

		int Height() const
		{
			return src.Height();
		}

		bool pixel(int x, int y) const
		{
			return !src.pixel(x, y);
		}
	};

	template <class SourceImageType, class DestinationImageType>
	void Erosion(const ImageReadable<bool, SourceImageType> &src, ImageWritable<bool, DestinationImageType> &dst, float rad)
	{
		Image<int> tmp(src.Width(), src.Height());
		EDT::Simple(InvertImage<SourceImageType>(src), tmp);
		
		int dthr = (int)(rad * rad);

		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				dst(i, j) = (tmp(i, j) > dthr || tmp(i, j) == -1);
	}

	template <class SourceImageType, class DestinationImageType>
	void Dilation(const ImageReadable<bool, SourceImageType> &src, ImageWritable<bool, DestinationImageType> &dst, float rad)
	{
		Image<int> tmp(src.Width(), src.Height());
		EDT::Simple(src, tmp);

		int dthr = (int)(rad * rad);

		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				dst(i, j) = (tmp(i, j) <= dthr);
	}

}
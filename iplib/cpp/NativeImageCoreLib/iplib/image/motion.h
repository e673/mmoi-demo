#pragma once

#include "core.h"
#include <algorithm>

namespace ip
{
	template <typename SourcePixelType, class SourceImageType, class VectorImageType, typename DestinationImageType>
	void ForwardMotionWarping(const ImageReadable<SourcePixelType, SourceImageType> &src, const ImageReadable<PixelFloatVector, VectorImageType> &vec,
		ImageWritable<SourcePixelType, DestinationImageType> &res, float sigma, float factor)
	{
		float rad = 5.0f * sigma;
		float t = -1.0f / (2.0f * sigma * sigma);

		ImageFloat w(src.Width(), src.Height());

		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				res(i, j) = SourcePixelType();
				w(i, j) = 0.0f;
			}
		}

		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				PixelFloatVector p = vec(i, j);
				float x = i + p.x * factor;
				float y = j + p.y * factor;

				SourcePixelType v = src(i, j);

				int x1 = (std::max)(0, (int)(x - rad));
				int x2 = (std::min)(src.Width() - 1, (int)(x + rad + 1.0f));
				int y1 = (std::max)(0, (int)(y - rad));
				int y2 = (std::min)(src.Height() - 1, (int)(y + rad + 1.0f));

				for (int yy = y1; yy <= y2; yy++)
				{
					for (int xx = x1; xx <= x2; xx++)
					{
						float qx = xx - x, qy = yy - y;
						float q = expf((qx * qx + qy * qy) * t);
						w(xx, yy) += q;
						res(xx, yy) += v * q;
					}
				}
			}
		}

		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				res(i, j) *= (1.0f / w(i, j));
	}
}
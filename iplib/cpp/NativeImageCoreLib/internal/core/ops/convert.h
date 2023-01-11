#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "../base/imagebase.h"

namespace ip
{
	template <typename DestinationPixelType, typename SourcePixelType>
	class OperationConvert
	{
	public:
		DestinationPixelType operator () (SourcePixelType src) const
		{
			return (DestinationPixelType)src;
		}

		typedef DestinationPixelType PixelType;
	};
}
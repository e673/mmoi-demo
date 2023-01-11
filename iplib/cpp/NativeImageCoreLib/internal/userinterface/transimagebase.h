#pragma once

#include "../../iplib/image/core.h"
#include "../../iplib/image/io/dibitmap.h"

namespace ip
{
	template <class BitmapDataStructureType>
	class TransImageBase
	{
	protected:
		BitmapDataStructureType data;
		std::shared_ptr<internal::BitmapDataResource<BitmapDataStructureType>> base_data;
	};
}
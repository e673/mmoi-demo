#pragma once

#include <vector>
#include <string>
#include "../../iplib/image/core.h"
#include "transimage.h"

namespace ip
{
	namespace internal
	{
		template <typename T>
		struct UIPixelID
		{
			static const int ID = 0;
		};

		template <>
		struct UIPixelID < float >
		{
			static const int ID = 1;
		};

		class UIDisplay
		{
			std::wstring title;
			std::vector<TransImage*> images;
			std::vector<std::wstring> names;
			std::vector<int> types;

		public:
			UIDisplay(std::wstring title)
				: title(std::move(title)) {}
			
			template <class PixelType, class ImageType>
			UIDisplay& AddImage(const ImageReadable<PixelType, ImageType> &image, std::wstring name)
			{
				TransImage *img = new TransImage();

				img->CreateImage(image.Width(), image.Height(), sizeof(PixelType));

				for (int j = 0; j < image.Height(); j++)
					for (int i = 0; i < image.Width(); i++)
						((Image<PixelType>*)img)->pixel(i, j) = image.pixel(i, j);

				images.push_back(img);
				names.emplace_back(std::move(name));
				types.push_back(UIPixelID<PixelType>::ID);

				return *this;
			}

			~UIDisplay();
		};
	}
}
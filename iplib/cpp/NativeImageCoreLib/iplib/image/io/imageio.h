#pragma once

#include <Windows.h>
#include <GdiPlus.h>

#include "../core.h"

namespace ip
{
	namespace internal
	{
		class LockedBitmap
		{
			Gdiplus::Bitmap &B;
			void Init(Gdiplus::PixelFormat pf, int pixelsize);

		public:
			Gdiplus::BitmapData bitmapdata;

			LockedBitmap(Gdiplus::Bitmap &B, Gdiplus::PixelFormat pf, int pixelsize);
			LockedBitmap(Gdiplus::Bitmap &B);
			~LockedBitmap();
		};

		class GdiPlusInitializer
		{
			ULONG_PTR m_gdiplusToken;
			static GdiPlusInitializer single_instance;

		public:
			GdiPlusInitializer();
			~GdiPlusInitializer();
		};
	}

	//////////////////////////////////////////////////////////////////////////

	class ImageIO
	{
	private:
		static int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);

	public:

		template <typename PixelType>
		static Image<PixelType> FromBitmap(Gdiplus::Bitmap &B)
		{
			int Width = B.GetWidth(), Height = B.GetHeight();
			Image<PixelType> res(Width, Height);

			if (B.GetPixelFormat() == PixelFormat8bppIndexed)
			{
				int pal_size = B.GetPaletteSize();	// Get palette size in bytes
				Gdiplus::ColorPalette *palette = (Gdiplus::ColorPalette*)malloc(pal_size);
				B.GetPalette(palette, pal_size);

				assert(palette != nullptr && palette->Entries != nullptr);

				internal::LockedBitmap lbi(B, PixelFormat8bppIndexed, 1);
				CustomBitmapImage<unsigned char> src;
				src.Init(lbi.bitmapdata.Scan0, Width, Height, lbi.bitmapdata.Stride);

				for (int j = 0; j < Height; j++)
				{
					CustomBitmapImage<unsigned char>::PixelReader reader = src.GetPixelReadIterator(0, j);
					Image<PixelType>::PixelWriter writer = res.GetPixelWriteIterator(0, j);
					for (int i = 0; i < Width; i++)
					{
						int c = reader.get();
						if (c < 0 && c >= (int)palette->Count)
							continue;

						Gdiplus::ARGB color = palette->Entries[c];
						PixelByteRGB p;
						p.b = (color & 0xFF);
						p.g = ((color >> 8) & 0xFF);
						p.r = ((color >> 16) & 0xFF);

						writer.set((PixelType)p);

						reader.advance(1);
						writer.advance(1);
					}
				}

				free(palette);
			}
			else
			{
				internal::LockedBitmap lbi(B);
				CustomBitmapImage<PixelByteRGB> src;
				src.Init(lbi.bitmapdata.Scan0, Width, Height, lbi.bitmapdata.Stride);

				for (int j = 0; j < Height; j++)
					for (int i = 0; i < Width; i++)
						res(i, j) = (PixelType)src(i, j);
			}

			return res;
		}

		template <typename PixelType>
		static Image<PixelType> FromFile(const WCHAR* filename)
		{
			Gdiplus::Bitmap B(filename);
			if (B.GetLastStatus() != 0)
				return Image<PixelType>();
			else
				return FromBitmap<PixelType>(B);
		}

		template <typename PixelType, class ImageType>
		static Gdiplus::Bitmap* ToBitmap(const ImageBase<PixelType, ImageType>& image)
		{
			Gdiplus::Bitmap *B = new Gdiplus::Bitmap(image.Width(), image.Height(), PixelFormat24bppRGB);
			internal::LockedBitmap lbi(*B);
			CustomBitmapImage<PixelByteRGB> dst;
			dst.Init(lbi.bitmapdata.Scan0, image.Width(), image.Height(), lbi.bitmapdata.Stride);
			// dst.Nya();
			image.Convert<PixelByteRGB>().CopyTo(dst);
			return B;
		}

		template <typename PixelType, class ImageType>
		static void ToFile(const ImageReadable<PixelType, ImageType>& image, const WCHAR* filename)
		{
			std::unique_ptr<Gdiplus::Bitmap> B(ToBitmap(image));
			CLSID pngClsid;
			GetEncoderClsid(L"image/png", &pngClsid);
			B->Save(filename, &pngClsid);
		}
	};

}


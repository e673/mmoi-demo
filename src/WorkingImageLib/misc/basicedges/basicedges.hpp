#pragma once

#include <iplib/image/core.h>
#include <iplib/image/canny.h>
#include <iplib/image/morphology/binarymorphology.h>
#include <iplib/image/transform.h>
#include <iplib/image/analysis/objectdetection.h>

namespace ip
{
	class BasicEdges
	{
	public:
		template <typename PixelType, class ImageType>
		BasicEdges(const ImageReadable<PixelType, ImageType> &src, float half_edge_width, float edge_influence_factor = 3.0f, float min_edge_distance_factor = 2.0f);

		ImageFloatColor4 Visualize();

		template <typename PixelType, class ImageType>
		static ImageFloat CalculateEdgePowerMask(const ImageReadable<PixelType, ImageType> &src, float edge_width);

	private:
		static constexpr float EP_Const = 0.5f;
		static constexpr float Grad_Thr = 0.001f;	// 0.1f default
		static constexpr int MinPixels = 20;

		ImageBinary bepmask, benmask, befmask, bamask;
		ImageFloat fedges;     // Фильтрованные эджи
		ImageFloat sedges;     // Фильтрованные эджи, прошедшие порог по градиенту
		ImageFloat edgepowermask;

		static void MaxFilter(float *src, int N, float sigma);

		template <class ImageType>
		static void RefineBasicEdges(ImageWritable<float, ImageType> &sedges);
	};

	template <class ImageType>
	void BasicEdges::RefineBasicEdges(ImageWritable<float, ImageType> &sedges)
	{
		ObjectDetection res = ObjectDetection(PixelTransform(sedges, [](float v){ return v > 0.0f; }), true);

		for (int j = 0; j < sedges.Height(); j++)
			for (int i = 0; i < sedges.Width(); i++)
				if (res(i, j) >= 0 && res.GetObjectInfo(res(i, j)).NumPixels < MinPixels)
					sedges(i, j) = 0.0f;
	}

	template <typename PixelType, class ImageType>
	BasicEdges::BasicEdges(const ImageReadable<PixelType, ImageType> &src, float edge_width, float edge_influence_factor, float min_edge_distance_factor)
		: bepmask(src.Width(), src.Height())
		, benmask(src.Width(), src.Height())
		, befmask(src.Width(), src.Height())
		, bamask(src.Width(), src.Height())
		, fedges(src.Width(), src.Height())
		, sedges(src.Width(), src.Height())
	{
		float r = edge_width;
		float R = edge_influence_factor * edge_width;
		float rT = min_edge_distance_factor * edge_width;

		float r2 = r * r;
		float R2 = R * R;

		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				bepmask(i, j) = benmask(i, j) = befmask(i, j) = bamask(i, j) = false;				

		Canny edges(src, edge_width * 0.5f);

		float maxedgepower = 0.0f;
		for (int j = 0; j < edges.Height(); j++)
			for (int i = 0; i < edges.Width(); i++)
				if (edges(i, j) > maxedgepower)
					maxedgepower = edges(i, j);

		// Считаем Edge Power Mask
		edgepowermask.swap(CalculateEdgePowerMask(edges, 2.0f * edge_width));

		// И накладываем эту маску - в итоге остаются только немаскированные эджи
		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				fedges(i, j) = edges(i, j) > edgepowermask(i, j) * EP_Const ? edges(i, j) : 0.0f;
		
		// Теперь выделяем кандидаты в базовые эджи - условия с двух сторон
		ImageBinary mask1(src.Width(), src.Height());
		Erosion(PixelTransform(fedges, [](float x) { return (x == 0.0f); }), mask1, rT);
		Dilation(mask1, mask1, rT + 1.0f);
		Erosion(mask1, mask1, 2.0f);

		// ... и с одной стороны
		ImageBinary mask2(src.Width(), src.Height());
		Erosion(PixelTransform(fedges, [](float x) { return (x == 0.0f); }), mask2, R);
		Dilation(mask2, mask2, R + 1.0f);

		// Базовые эджи - эджи, прошедшие по порогу и попавшие в обе маски
		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				sedges(i, j) = (fedges(i, j) > maxedgepower * Grad_Thr && mask1(i, j) && mask2(i, j)) ? fedges(i, j) : 0.0f;

		// Исключаем концевые точки
		for (int j = 1; j < src.Height() - 1; j++)
		{
			for (int i = 1; i < src.Width() - 1; i++)
			{
				// Только базовые эджи
				if (sedges(i, j) == 0.0f)
					continue;

				int nbord = 0x00;
				if (fedges(i - 1, j - 1) > 0.0f)
					nbord |= 0x01;
				if (fedges(i, j - 1) > 0.0f)
					nbord |= 0x02;
				if (fedges(i + 1, j - 1) > 0.0f)
					nbord |= 0x04;
				if (fedges(i + 1, j) > 0.0f)
					nbord |= 0x08;
				if (fedges(i + 1, j + 1) > 0.0f)
					nbord |= 0x10;
				if (fedges(i, j + 1) > 0.0f)
					nbord |= 0x20;
				if (fedges(i - 1, j + 1) > 0.0f)
					nbord |= 0x40;
				if (fedges(i - 1, j) > 0.0f)
					nbord |= 0x80;

				// Не трогаем концы эджей
				if (nbord == 0x00 ||
					nbord == 0x01 || nbord == 0x02 || nbord == 0x04 || nbord == 0x08 || nbord == 0x10 || nbord == 0x20 || nbord == 0x40 || nbord == 0x80 ||
					nbord == 0x03 || nbord == 0x06 || nbord == 0x0C || nbord == 0x18 || nbord == 0x30 || nbord == 0x60 || nbord == 0xC0 || nbord == 0x81)
				{
					sedges(i, j) = 0.0f;
				}
			}
		}

		// Сужаем область базовых контуров
		ImageBinary mask3(src.Width(), src.Height());
		Dilation(PixelTransform(fedges, sedges, [](float vf, float vs) { return vf > 0.0f && vs == 0.0f; }), mask3, edge_width);
		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				if (mask3(i, j))
					sedges(i, j) = 0.0f;

		// Задаём минимальное количество точек в базовом контуре
		RefineBasicEdges(sedges);

		// Вычисляем расстояния от найденных базовых контуров
		Image<int> edge_dist(src.Width(), src.Height());
		Image<PixelVector<int>> edge_ref(src.Width(), src.Height());
		EDT::Extended(PixelTransform(fedges, [](float x) { return x > 0.0f; }), edge_dist, edge_ref);

		// Заполняем bamask и befmask - область влияния базовых контуров
		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				bamask(i, j) = sedges(edge_ref(i, j).x, edge_ref(i, j).y) > 0.0f;
				befmask(i, j) = bamask(i, j) && (edge_dist(i, j) > R2);
			}
		}

		// Вычисляем расстояния от границы зоны влияния контуров
		Image<int> border_dist(src.Width(), src.Height());
		EDT::Simple(PixelTransform(edge_dist, bamask, [R2](int d, bool b) { return b && (d > R2); }), border_dist);

		// Теперь создаём маски
		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				if (!bamask(i, j))
					continue;

				int d1 = edge_dist(i, j);
				int d2 = border_dist(i, j);

				if (sqrtf((float)d1) + sqrtf((float)d2) <= R + 2.0f)
				{
					if (d1 <= r2)
						bepmask(i, j) = true;
					else if (d1 <= R2)
						benmask(i, j) = true;
				}
			}
		}
	}

	template <typename PixelType, class ImageType>
	ImageFloat BasicEdges::CalculateEdgePowerMask(const ImageReadable<PixelType, ImageType> &src, float edge_width)
	{
		ImageFloat res(src.Width(), src.Height());


		float *buf = new float[src.Width() + src.Height()];

		// Horizontal pass
		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = 0; i < src.Width(); i++)
				buf[i] = src(i, j);

			MaxFilter(buf, src.Width(), edge_width);

			for (int i = 0; i < src.Width(); i++)
				res(i, j) = buf[i];
		}

		// Vertical pass
		for (int i = 0; i < src.Width(); i++)
		{
			for (int j = 0; j < src.Height(); j++)
				buf[j] = res(i, j);

			MaxFilter(buf, src.Height(), edge_width);

			for (int j = 0; j < src.Height(); j++)
				res(i, j) = buf[j];
		}

		delete[] buf;

		return res;
	}

	void BasicEdges::MaxFilter(float *src, int N, float sigma)
	{
		float mult = expf(-1.0f / (2.0f * sigma * sigma));
		float *fwd = new float[N], *bkw = new float[N];

		float v = 0.0f;
		for (int i = 0; i < N; i++)
		{
			v *= mult;
			if (src[i] > v)
				v = src[i];
			fwd[i] = v;
		}

		v = 0.0f;
		for (int i = N - 1; i >= 0; i--)
		{
			v *= mult;
			if (src[i] > v)
				v = src[i];
			bkw[i] = v;
		}

		for (int i = 0; i < N; i++)
		{
			src[i] = fwd[i] > bkw[i] ? fwd[i] : bkw[i];;
		}

		delete[] fwd;
		delete[] bkw;
	}


	ImageFloatColor4 BasicEdges::Visualize()
	{
		ImageFloatColor4 res(fedges.Width(), fedges.Height());

		for (int j = 0; j < res.Height(); j++)
		{
			for (int i = 0; i < res.Width(); i++)
			{
				if (sedges(i, j) > 0.0f)
					res(i, j) = PixelFloatRGB4(255, 255, 255);
				else if (fedges(i, j) > 0.0f)
					res(i, j) = PixelFloatRGB4(128, 128, 128);
				else if (bepmask(i, j))
					res(i, j) = PixelFloatRGB4(255, 192, 0);
				else if (benmask(i, j))
					res(i, j) = PixelFloatRGB4(0, 192, 0);
				else if (befmask(i, j))
					res(i, j) = PixelFloatRGB4(0, 0, 96);

				/* else if (bamask(i, j))
					res(i, j) = PixelFloatRGB4(255, 0, 255); */

				//if (fedges(i, j) > edgepowermask(i, j) * 0.5f)
				//	res(i, j) = PixelFloatRGB4(0.0f, fedges(i, j) * 16.0f, 0.0f);

			}
		}

		return res;
	}
}

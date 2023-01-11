#pragma once

#include <vector>
#include "../core.h"
#include "../morphology/binarymorphology.h"

namespace ip
{
	struct ObjectDetectionInfo
	{
		int NumPixels;
		int x0, x1;  // Inclusive
		int y0, y1;  // Inclusive
		float center_x, center_y;
	};

	// ==================================================================================================

	class ObjectDetection
		: public Image<int>
	{
	private:
		class LinkedChainBase;

	public:
		template <class ImageType>
		ObjectDetection(const ImageReadable<bool, ImageType> &src, bool DiagLinks);

		int GetObjectCount();
		int GetBackgroundCount();
		const ObjectDetectionInfo& GetObjectInfo(int index);
		const ObjectDetectionInfo& GetBackgroundInfo(int index);

	private:
		int objectCount, backgroundCount;
		std::vector<ObjectDetectionInfo> objectInfo, backgroundInfo;

		template <class ImageType>
		int DetectForegroundObjects(const ImageReadable<bool, ImageType> &src, bool DiagLinks);

		template <class ImageType>
		int DetectBackgroundObjects(const ImageReadable<bool, ImageType> &src, bool DiagLinks);

		template <class ImageType>
		int DetectSubObjects(const ImageReadable<bool, ImageType> &src, Image<int> &res, bool DiagLinks);

		void AnalyzeObjects();

		static void PrepareMask(Image<bool> &img);
	};

	// ==================================================================================================

	class ObjectDetection::LinkedChainBase
	{
		std::vector<int> links;

	public:
		int GetNextLink();
		int GetBaseNode(int id);
		int GetNode(int id);
		int Connect(int id1, int id2);
		int LinkAndEnumerate();
	};

	// ==================================================================================================

	template <class ImageType>
	ObjectDetection::ObjectDetection(const ImageReadable<bool, ImageType> &src, bool DiagLinks)
		: Image<int>(src.Width(), src.Height())
	{
		objectCount = DetectForegroundObjects(src, DiagLinks);
		backgroundCount = DetectBackgroundObjects(src, DiagLinks);
		AnalyzeObjects();
	}

	template <class ImageType>
	int ObjectDetection::DetectForegroundObjects(const ImageReadable<bool, ImageType> &src, bool DiagLinks)
	{
		Image<int> fg_obj(Width(), Height());
		int cnt = DetectSubObjects(src, fg_obj, DiagLinks);

		for (int j = 0; j < Height(); j++)
			for (int i = 0; i < Width(); i++)
				if (fg_obj(i, j) >= 0)
					pixel(i, j) = fg_obj(i, j);

		return cnt;
	}

	template <class ImageType>
	int ObjectDetection::DetectBackgroundObjects(const ImageReadable<bool, ImageType> &src, bool DiagLinks)
	{
		Image<int> bg_obj(Width(), Height());
		int cnt = DetectSubObjects(InvertImage<ImageType>(src), bg_obj, !DiagLinks);

		for (int j = 0; j < Height(); j++)
			for (int i = 0; i < Width(); i++)
				if (bg_obj(i, j) >= 0)
					pixel(i, j) = -bg_obj(i, j) - 1;

		return cnt;
	}

	/// <summary>
	/// Detect only foreground objects
	/// </summary>
	template <class ImageType>
	int ObjectDetection::DetectSubObjects(const ImageReadable<bool, ImageType> &src, Image<int> &res, bool DiagLinks)
	{
		check(src.Width() == res.Width() && src.Height() == res.Height());

		LinkedChainBase links;
		std::unique_ptr<bool[]> curline(new bool[src.Width() + 2]);
		std::unique_ptr<bool[]> prevline(new bool[src.Width() + 2]);

		for (int i = 0; i < src.Width() + 2; i++)
			prevline[i] = false;

		for (int j = 0; j < src.Height(); j++)
		{
			curline[0] = false;
			curline[src.Width() + 1] = false;
			for (int i = 0; i < src.Width(); i++)
				curline[i + 1] = src(i, j);

			for (int i = 0; i < src.Width(); i++)
			{
				res(i, j) = -1;

				if (curline[i + 1])
				{
					if (curline[i])
						res(i, j) = links.Connect(res(i, j), res(i - 1, j));

					if (prevline[i + 1])
						res(i, j) = links.Connect(res(i, j), res(i, j - 1));

					if (prevline[i] && DiagLinks)
						res(i, j) = links.Connect(res(i, j), res(i - 1, j - 1));

					if (prevline[j + 2] && DiagLinks)
						res(i, j) = links.Connect(res(i, j), res(i + 1, j - 1));

					if (res(i, j) == -1)
						res(i, j) = links.GetNextLink();
				}
			}

			std::swap(curline, prevline);
		}

		int cnt = links.LinkAndEnumerate();

		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				if (res(i, j) != -1)
					res(i, j) = links.GetNode(res(i, j));

		return cnt;
	}
}
#include "../../iplib/image/analysis/objectdetection.h"

namespace ip
{
	int ObjectDetection::LinkedChainBase::GetNextLink()
	{
		int nextval = (int)links.size();
		links.push_back(nextval);
		return nextval;
	}

	int ObjectDetection::LinkedChainBase::GetBaseNode(int id)
	{
		// This node is already base node
		int id1 = links[id];
		if (id1 == id)
			return id;

		// The linked node is already a base node
		if (links[id1] == id1)
			return id1;

		// The linked node is not a base node => will perform optimization
		while (id1 != links[id1])
			id1 = links[id1];

		// Now perform tree optimization
		while (id != id1)
		{
			int id2 = links[id];
			links[id] = id1;
			id = id2;
		}

		return id1;
	}

	int ObjectDetection::LinkedChainBase::GetNode(int id)
	{
		return links[id];
	}

	int ObjectDetection::LinkedChainBase::Connect(int id1, int id2)
	{
		if (id1 == -1)
			return id2;

		if (id2 == -1)
			return id1;

		int rid1 = GetBaseNode(id1);
		int rid2 = GetBaseNode(id2);

		links[id1] = rid1;
		links[id2] = rid2;

		if (rid1 < rid2)
		{
			links[rid2] = rid1;
			return rid1;
		}
		else if (rid1 > rid2)
		{
			links[rid1] = rid2;
			return rid2;
		}
		else
			return rid1;
	}

	/// <summary>
	/// Link all segments
	/// </summary>
	/// <returns>Total number of different objects</returns>
	int ObjectDetection::LinkedChainBase::LinkAndEnumerate()
	{
		std::vector<int> rlinks(links.size());

		int res = 0;

		for (int i = 0; i < (int)links.size(); i++)
		{
			links[i] = GetBaseNode(i);
			if (links[i] == i)
				rlinks[i] = res++;
		}

		for (int i = 0; i < (int)links.size(); i++)
			links[i] = rlinks[links[i]];

		return res;
	}

	// ==================================================================================================

	void ObjectDetection::PrepareMask(Image<bool> &mask)
	{
		for (int j = 1; j < mask.Height() - 1; j++)
			for (int i = 1; i < mask.Width() - 1; i++)
				if (mask(i, j - 1) && mask(i, j + 1) && mask(i - 1, j) && mask(i + 1, j) && !mask(i, j))
					mask(i, j) = true;
	}

	void ObjectDetection::AnalyzeObjects()
	{
		objectInfo = std::vector<ObjectDetectionInfo>(objectCount, ObjectDetectionInfo());
		backgroundInfo = std::vector<ObjectDetectionInfo>(backgroundCount, ObjectDetectionInfo());

		for (int y = 0; y < Height(); y++)
		{
			for (int x = 0; x < Width(); x++)
			{
				int v = pixel(x, y);

				ObjectDetectionInfo &oinfo = v >= 0 ? objectInfo[v] : backgroundInfo[-v - 1];

				if (oinfo.NumPixels == 0)
				{
					oinfo.x0 = oinfo.x1 = x;
					oinfo.y0 = oinfo.y1 = y;
				}
				else
				{
					if (x < oinfo.x0)
						oinfo.x0 = x;

					if (x > oinfo.x1)
						oinfo.x1 = x;

					if (y < oinfo.y0)
						oinfo.y0 = y;

					if (y > oinfo.y1)
						oinfo.y1 = y;
				}

				oinfo.NumPixels++;
				oinfo.center_x += x;
				oinfo.center_y += y;
			}
		}

		for (ObjectDetectionInfo &oinfo : objectInfo)
		{
			oinfo.center_x /= oinfo.NumPixels;
			oinfo.center_y /= oinfo.NumPixels;
		}

		for (ObjectDetectionInfo &oinfo : backgroundInfo)
		{
			oinfo.center_x /= oinfo.NumPixels;
			oinfo.center_y /= oinfo.NumPixels;
		}
	}

	int ObjectDetection::GetObjectCount()
	{
		return objectCount;
	}

	int ObjectDetection::GetBackgroundCount()
	{
		return backgroundCount;
	}

	const ObjectDetectionInfo& ObjectDetection::GetObjectInfo(int index)
	{
		return objectInfo[index];
	}

	const ObjectDetectionInfo& ObjectDetection::GetBackgroundInfo(int index)
	{
		return backgroundInfo[index];
	}

}
#include "../../iplib/userinterface.h"

#include "uidisplayaccumulator.h"
#include <array>
#include <algorithm>

#include "transimage.h"
#include "transimage3d.h"

namespace ip
{
	namespace internal
	{
		UIDisplay::~UIDisplay()
		{
			UI::DisplayImages(images, names, types, title);
		}
	}

	std::unique_ptr<UserInterfaceFuncData> UI::uifuncdata;

	void UI::Reset()
	{
		if (uifuncdata)
			uifuncdata->Reset();
	}

	void UI::ReportProgress(float ProgressValue)
	{
		if (uifuncdata)
			uifuncdata->ReportProgress(ProgressValue, nullptr);
	}

	void UI::ReportProgress(float ProgressValue, const std::wstring &Text)
	{
		if (uifuncdata)
			uifuncdata->ReportProgress(ProgressValue, Text.c_str());
	}

	void UI::ReportProgressText(const std::wstring &Text)
	{
		if (uifuncdata)
			uifuncdata->ReportProgressText(Text.c_str());
	}

	void UI::ReportGlobalText(const std::wstring &Text)
	{
		if (uifuncdata)
			uifuncdata->ReportGlobalText(Text.c_str());
	}

	void UI::BeginRange(float ProgressRange, bool SubLevel)
	{
		if (uifuncdata)
			uifuncdata->BeginRange(ProgressRange, false, SubLevel);
	}

	void UI::BeginRange(float ProgressRange, const std::wstring &Text, bool SubLevel)
	{
		if (uifuncdata)
		{
			uifuncdata->BeginRange(ProgressRange, false, SubLevel);
			uifuncdata->ReportProgressText(Text.c_str());
		}
	}

	void UI::BeginSubRange(float ProgressRange)
	{
		if (uifuncdata)
		{
			uifuncdata->BeginRange(ProgressRange, true, true);
		}
	}

	void UI::BeginSubRange(float ProgressRange, const std::wstring &Text)
	{
		if (uifuncdata)
		{
			uifuncdata->BeginRange(ProgressRange, true, true);
			uifuncdata->ReportProgressText(Text.c_str());
		}
	}

	void UI::EndRange()
	{
		if (uifuncdata)
			uifuncdata->EndRange();
	}

	void UI::BeginIterationSet(float ProgressRange, int NumIterations, const std::wstring &Text, bool SubProgressBar)
	{
		if (uifuncdata)
		{
			uifuncdata->BeginRange(ProgressRange, SubProgressBar, true);
			uifuncdata->BeginIterationRange(NumIterations, Text.c_str());
		}
	}

	void UI::BeginIterationSet(int NumIterations, const std::wstring &Text, bool SubProgressBar)
	{
		if (uifuncdata)
		{
			uifuncdata->BeginRange(1.0f, SubProgressBar, true);
			uifuncdata->BeginIterationRange(NumIterations, Text.c_str());
		}
	}

	void UI::ReportIteration(int Iteration)
	{
		if (uifuncdata)
			uifuncdata->ReportIteration(Iteration, nullptr);
	}

	void UI::ReportIteration(int Iteration, const std::wstring &Text)
	{
		if (uifuncdata)
			uifuncdata->ReportIteration(Iteration, Text.c_str());
	}

	void UI::EndIterationSet()
	{
		if (uifuncdata)
		{
			uifuncdata->EndRange();
			uifuncdata->EndRange();
		}
	}

	internal::UIDisplay UI::Display(std::wstring title)
	{
		return internal::UIDisplay(std::move(title));
	}

	void UI::DisplayImages(const std::vector<TransImage*> &images, const std::vector<std::wstring> &names, const std::vector<int> &types, const std::wstring &title)
	{
		if (uifuncdata)
		{
			std::vector<TransImage*> v_images(images.size());
			std::vector<const wchar_t*> v_names(names.size());
			std::transform(names.begin(), names.end(), v_names.begin(), [](const std::wstring &v) { return v.c_str(); });
			std::transform(images.begin(), images.end(), v_images.begin(), [](TransImage *img) { return img; });
			uifuncdata->DisplayImages(&v_images[0], &v_names[0], types.data(), (int)images.size(), title.c_str());
		}
	}

	bool UI::StopRequested()
	{
		if (uifuncdata)
			return uifuncdata->StopRequested();
		else
			return false;
	}

	void UI::ForceUIRepaint()
	{
		if (uifuncdata)
			uifuncdata->ForceUIRepaint();
	}

	void UI::ConfigureUI(const UserInterfaceFuncData &uifuncdata)
	{
		UI::uifuncdata.reset(new UserInterfaceFuncData(uifuncdata));
	}

	// ==================================================================================================

	UserInterfaceFuncData::~UserInterfaceFuncData()
	{
		// DestroyHandle(GCHandle);
	}

	// ==================================================================================================
	//                                          TransImage               
	// ==================================================================================================

	void TransImage::CreateDIBitmap(int Width, int Height, HDC *hdc)
	{
		internal::DIBitmapData *data = new internal::DIBitmapData(Width, Height);
		// *hbitmap = data->hbitmap;
		*hdc = data->hdc;
		base_data.reset(data);
		this->data = *base_data;
	}

	void TransImage::CreateImage(int Width, int Height, int PixelSize)
	{
		base_data.reset(new internal::BitmapData(Width, Height, PixelSize));
		data = *base_data;
	}

	void TransImage::CreateFragment(TransImage &img, int OfsX, int OfsY, int Width, int Height, int PixelSize)
	{
		data = img.base_data->Fragment(img.data, OfsX, OfsY, Width, Height, PixelSize);
		base_data = img.base_data;
	}

	void TransImage3D::CreateImage3D(int SizeX, int SizeY, int SizeZ, int PixelSize)
	{
		base_data.reset(new internal::BitmapData3D(SizeX, SizeY, SizeZ, PixelSize));
		data = *base_data;
	}

	void TransImage3D::CreateFragment(TransImage3D &img, int OfsX, int OfsY, int OfsZ, int SizeX, int SizeY, int SizeZ, int PixelSize)
	{
		data = img.base_data->Fragment(img.data, OfsX, OfsY, OfsZ, SizeX, SizeY, SizeZ, PixelSize);
		base_data = img.base_data;
	}
}

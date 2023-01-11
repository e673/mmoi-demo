#pragma once

#include "image/core.h"
#include <string>
#include <vector>

#include "../internal/userinterface/uidisplayaccumulator.h"
#include "../internal/userinterface/userinterfacefuncdata.h"

namespace ip
{
	class UI
	{
		static std::unique_ptr<UserInterfaceFuncData> uifuncdata;

	public:
		static void Reset();
		static void ReportProgress(float ProgressValue);
		static void ReportProgress(float ProgressValue, const std::wstring &Text);
		static void ReportProgressText(const std::wstring &Text);
		static void ReportGlobalText(const std::wstring &Text);
		static void BeginRange(float ProgressRange, bool SubLevel = false);
		static void BeginRange(float ProgressRange, const std::wstring &Text, bool SubLevel = false);
		static void BeginSubRange(float ProgressRange);
		static void BeginSubRange(float ProgressRange, const std::wstring &Text);
		static void EndRange();
		static void BeginIterationSet(float ProgressRange, int NumIterations, const std::wstring &Text, bool SubProgressBar = false);
		static void BeginIterationSet(int NumIterations, const std::wstring &Text, bool SubProgressBar = false);
		static void ReportIteration(int Iteration);
		static void ReportIteration(int Iteration, const std::wstring &Text);
		static void EndIterationSet();
		static internal::UIDisplay Display(std::wstring title);
		static void DisplayImages(const std::vector<TransImage*> &images, const std::vector<std::wstring> &names, const std::vector<int> &types, const std::wstring &title);
		static bool StopRequested();
		static void ForceUIRepaint();

		static void ConfigureUI(const UserInterfaceFuncData &uifuncdata);
	};
}
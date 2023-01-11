#pragma once

#include "transimage.h"
#include "transimage3d.h"

namespace ip
{
	#pragma pack(push, 1)

	struct UserInterfaceFuncData
	{
		void *GCHandle;

		void (__stdcall *Reset)();
		void (__stdcall *ReportProgress)(float progress, const wchar_t *text);
		void (__stdcall *ReportProgressText)(const wchar_t *text);
		void (__stdcall *ReportGlobalText)(const wchar_t *text);
		void (__stdcall *BeginRange)(float progress_range, bool sub_progress_bar, bool add_hierarchy_level);
		void (__stdcall *EndRange)();
		void (__stdcall *BeginIterationRange)(int num_iter, const wchar_t *text);
		void (__stdcall *ReportIteration)(int iter, const wchar_t *text);
		void (__stdcall *DisplayImages)(TransImage **images, const wchar_t **names, const int *types, int N, const wchar_t *title);
		void (__stdcall *ForceUIRepaint)();
		bool (__stdcall *StopRequested)();
		void (__stdcall *DestroyHandle)(void *GCHandle);

		~UserInterfaceFuncData();
	};

	#pragma pack(pop)
}
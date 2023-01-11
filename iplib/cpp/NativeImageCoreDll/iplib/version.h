#pragma once

namespace ip
{
#ifdef IPLIBCORE_DLL
	#define DLLAPI __declspec(dllexport)
#else
	#define DLLAPI __declspec(dllimport)
#endif

	extern const char DLLAPI CompileDateTime[];
	extern const wchar_t DLLAPI CompileDateTimeW[];
}

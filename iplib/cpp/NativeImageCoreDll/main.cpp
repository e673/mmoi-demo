#include <Windows.h>

#include <internal/lazyinit_cpp.hpp>
#include <internal/parallel_cpp.hpp>
#include <internal/version_cpp.hpp>

#include <stdio.h>

BOOL WINAPI DllMain(HINSTANCE module_handle, DWORD reason_for_call, LPVOID reserved)
{
	// printf("DLLMain: %d", reason_for_call);

	if (reason_for_call == DLL_PROCESS_ATTACH) // Self-explanatory
	{
		// printf("Attach\n");
		/* ip::parallel();
		ip::parallel.reset(); */
	}

	if (reason_for_call == DLL_PROCESS_DETACH) // Self-explanatory
	{
		ip::parallel.reset();

		// ip::parallel.forget();

		// printf("Detach\n");
		/* ip::parallel.reset(); */
	}

	return TRUE;
}

#pragma once

// ==================================================================================================
//                                    DLL export macro
// ==================================================================================================

#define EXPORT(type) extern "C" __declspec(dllexport) type __stdcall

// ==================================================================================================
//                                   Static polymorphism
// ==================================================================================================

#define static_interface_check(BaseClass, DerivedClass, FuncName) \
	const bool over = std::is_same<decltype(&DerivedClass::FuncName), decltype(&BaseClass<DerivedClass>::FuncName)>::value; \
	static_assert(!over, "static polymorphic function was not overwritten."); 

// ==================================================================================================
//                                System specific definitions
// ==================================================================================================

#undef die

#if _WIN32 || _WIN64	// Check windows
#
#   include <assert.h>
#
#	define die(_Message) _wassert(_CRT_WIDE(_Message), _CRT_WIDE(__FILE__), __LINE__)
#
#	if _WIN64
#		define ENV64BIT
#	else
#		define ENV32BIT
#	endif
#elif  __GNUC__			// Check GCC
#
#	define die(_Message) ((void)0)
#
#	if __x86_64__ || __ppc64__
#		define ENV64BIT
#	else
#		define ENV32BIT
#	endif
#else					// Default = 32 bit
#
#   define die(_Message) ((void)0)
#
#	define ENV32BIT
#
#endif

// ==================================================================================================
//                                          Assertions
// ==================================================================================================

#undef dbgcheck
#undef rtcheck
#undef check

// Macro dbgcheck is used in DEBUG mode only
#ifndef DEBUG
#	define dbgcheck(_Expression) ((void)0)
#else
#	define dbgcheck(_Expression) ((_Expression) ? ((void)0) : die(#_Expression))
#endif

// Macro rtcheck is used in DEBUG and RELEASE modes
#ifdef FULLRELEASE
#	define rtcheck(_Expression) ((void)0)
#else
#	define rtcheck(_Expression) ((_Expression) ? ((void)0) : die(#_Expression))
#endif

// Macro check is used in all modes
#define check(_Expression) ((_Expression) ? ((void)0) : die(#_Expression))

// ==================================================================================================
//                                      Type aliases
// ==================================================================================================

// Type aliases
typedef unsigned char byte;
typedef long long int64;
typedef unsigned long long uint64;

// ==================================================================================================
//                                          Misc               
// ==================================================================================================

namespace ip
{
	// A class that ignores any assignment
	class Ignore
	{
	public:
		template <class Any>
		void operator = (const Any&) const {}
	};

	extern const Ignore ignore;

	constexpr int log_of_two(int N)
	{
		return (N > 1) ? (1 + log_of_two((N + 1) / 2)) : (N == 1 ? 0 : -1);
	}

	constexpr int power_of_two(int N)
	{
		return (N > 0) ? (2 * power_of_two(N - 1)) : (N == 0 ? 1 : 0);
	}

	constexpr int nearest_power_of_two(int N)
	{
		return power_of_two(log_of_two(N));
	}

	template <typename T>
	constexpr int aligned_type_size()
	{
		return nearest_power_of_two(sizeof(T));
	}

	constexpr bool is_power_of_two(int N)
	{
		return (N > 1) ? ((N % 2 == 0) && is_power_of_two(N / 2)) : (N == 1 ? true : false);
	}

	constexpr int int_gr_com_div(int A, int B)
	{
		return (A == B || B == 0) ? A : int_gr_com_div(B, A % B);
	}

	constexpr int gr_com_div(int A, int B)
	{
		return (A > 0 && B > 0) ? (A > B ? int_gr_com_div(A, B) : int_gr_com_div(B, A)) : 0;
	}

	constexpr int gr_com_div(int A, int B, int C)
	{
		return gr_com_div(gr_com_div(A, B), C);
	}

	constexpr int ls_com_mul(int A, int B)
	{
		return A * B / gr_com_div(A, B);
	}

	constexpr int ls_com_mul(int A, int B, int C)
	{
		return -1;
	}

	/* template <int BlockSize, int... PixelSize>
	struct cycle_size
	{
	public:
		static const int value = BlockSize / gr_com_div<BlockSize, PixelSize...>::value;
	}; */

}

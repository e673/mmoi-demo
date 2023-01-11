#pragma once

#include <functional>
#include <atomic>

#include "lazyinit.h"
#include <new>

namespace ip
{

#ifdef IPLIBCORE_DLL
	#define DLLAPI __declspec(dllexport)
#else
	#define DLLAPI 	__declspec(dllimport)
#endif

	class DLLAPI Parallel
	{
	public:
		static void Aggregate(std::function<void(void* state)> func, std::function<void(void* accumulator, void* state)> aggregator, void *target, size_t state_size);
		static void Do(std::function<void()> func);
		static void For(std::function<void(std::atomic_int &counter)> func, int initial = 0);
		static void For(int beginInclusive, int endExclusive, std::function<void(int y)> func);
		static void Reset();

	public:
		template <typename T>
		static T Do(std::function<T()> func, std::function<void(T&, const T&)> aggregator);

		template <typename T>
		static T For(std::function<T(std::atomic_int &counter)> func, std::function<void(T&, const T&)> aggregator, int initial = 0);

		template <typename T>
		static T For(int beginInclusive, int endExclusive, std::function<void(int y, T& state)> func, std::function<void(T&, const T&)> aggregator);
	};


	template <typename T>
	T Parallel::Do(std::function<T()> func, std::function<void(T&, const T&)> aggregator)
	{
		T result;

		Aggregate([&func](void* _state)
		{
			auto state = reinterpret_cast<T*>(_state);
			new (state) T;
			*state = func();		
		},
		[&aggregator](void* _acc, void* _state)
		{
			auto acc = reinterpret_cast<T*>(_acc);
			auto state = reinterpret_cast<T*>(_state);

			aggregator(*acc, *state);
			state->~T();
		},
		&result, sizeof(result));

		return result;
	}

	template <typename T>
	T Parallel::For(std::function<T(std::atomic_int& counter)> func, std::function<void(T&, const T&)> aggregator, int initial)
	{
		std::atomic_int counter;
		counter.store(initial);
		return Do([&counter, &func] { return func(counter); }, aggregator);
	}

	template <typename T>
	T Parallel::For(int beginInclusive, int endExclusive, std::function<void(int y, T& state)> func, std::function<void(T&, const T&)> aggregator)
	{
		std::atomic_int counter;
		counter.store(beginInclusive);

		return Do<T>([&counter, &func, endExclusive]()
		{
			auto state = T();
			for (int y = counter++; y < endExclusive; y = counter++)
			{
				func(y, state);
			}
			return state;

		}, aggregator);
	}
}
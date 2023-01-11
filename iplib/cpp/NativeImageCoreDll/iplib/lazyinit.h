#pragma once

#include <memory>
#include <type_traits>
#include <mutex>

namespace ip
{
	namespace internal
	{
		extern std::recursive_mutex lazyinit_mutex;
	}

	/* Initialize the class only when it is accessed
	* Memory is allocated dynamically */
	template <class T>
	class LazyInit
	{
		std::unique_ptr<T> ptr;

		void AutoInit()
		{
			// Only one thread can simultaneously perform lazy initializations
			std::lock_guard<std::recursive_mutex> ml(internal::lazyinit_mutex);

			// If already initialized => do not re-initialize
			if (!ptr.get())
				ptr.reset(new T());
		}

	public:
		// Manual initialization
		template <class ... Args>
		void operator () (Args&&... args)
		{
			// Only one thread can simultaneously perform lazy initializations
			std::lock_guard<std::recursive_mutex> ml(internal::lazyinit_mutex);

			// Cannot re-initialize it manually!
			// ToDo
			// check(!ptr);

			ptr.reset(new T(std::forward<Args>(args)...));
		}

		T* get()
		{
			if (!ptr)
				AutoInit();

			return ptr.get();
		}

		T* operator -> ()
		{
			return get();
		}

		T& operator * ()
		{
			return *get();
		}

		void reset()
		{
			std::lock_guard<std::recursive_mutex> ml(internal::lazyinit_mutex);
			ptr.reset();
		}

		void forget()
		{
			ptr.release();
		}
	};
}
#include "../iplib/lazyinit.h"

namespace ip
{
	namespace internal
	{
		std::recursive_mutex lazyinit_mutex;
	}
}
#include <concepts>		

// Dummy file

template <typename TPixel, typename TImage>
concept IImage = requires (TImage a)
{
	{ a.get() } -> std::same_as<TPixel>;
};

template <typename TImage>
requires IImage<TPixel, TImage>
void TestFunc(TImage& src)
{

}

struct C
{
	int get() { return 0; }
};


void Test()
{
	// TestFunc<int, C>(C{});

}

template<typename LikeWhat>
struct like
{
	LikeWhat value;

	explicit like(const LikeWhat& value)
		: value(value) {}

	operator LikeWhat()
	{
		return value;
	}

	void assign(const LikeWhat& newVal)
	{
		value = newVal;
	}
};

struct wheels_count : like<int>
{
	using like<int>::like;
};

void Test2()
{
	auto wc = wheels_count(3);
	wc = wheels_count(4);
}
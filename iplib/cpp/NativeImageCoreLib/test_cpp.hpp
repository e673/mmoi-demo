#include <iplib/image/core.h>
#include <iplib/common.h>
#include <iplib/parallel.h>
#include <iplib/userinterface.h>

#include <stdio.h>



using namespace ip;

class A
	: public ImageBase<int, A>
{
public:
	int Width() const { return 1;  }
	int Height() const { return 1; }
};

template <class Derived>
class I1
{
public:
	void i1f()
	{
		static_cast<Derived*>(this)->i1f();
	}

};

template <class Derived>
class I2
	: I1<Derived>
{
public:
	void i2f()
	{
		static_cast<Derived*>(this)->i2f();
	}

};

template <class Derived>
class I3
	: I1<Derived>
{
public:
	void i3f()
	{
		static_cast<Derived*>(this)->i3f();
	}

};

class B
	: public I1<B>
	, public I2<B>
	, public I3<B>
{
public:
	void i1f()
	{
		printf("i1f\n");
	}

	void i2f()
	{
		printf("i2f\n");
	}

	void i3f()
	{
		printf("i3f\n");
	}
};

template <class T>
void Test2(I2<T>& i2)
{
	i2.i2f();
}

class CC
{
public:
	CC()
	{
		printf("CC constructed\n");
	}

	~CC()
	{
		printf("CC destroyed\n");
	}

	CC& AddValue(int v)
	{
		printf("Value added\n");
		return *this;
	}
};

CC CreateCC()
{
	return CC();
}

#include <iplib/math/matrix.h>

void PrintMatrix(const Matrix<double> & A)
{
	for (int i = 0; i < A.NumRows(); i++)
	{
		for (int j = 0; j < A.NumColumns(); j++)
		{
			printf("%9.1e", A(i, j));
		}

		printf("\n");
	}

	printf("\n");
}

void TestMatrix()
{
	int N = 12;
	Matrix<double> A(N, N);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A(i, j) = 1.0 / (i + j + 1.0);

	Matrix<double> I = Matrix<double>::Identity(N);

	Matrix<double> r1 = A.GaussSolve(I);
	Matrix<double> r2 = A.RotateSolve(I);

	PrintMatrix(r1);
	PrintMatrix(r2);
	PrintMatrix(A * r1);
	PrintMatrix(A * r2);
}

class TestClass
{
public:
	TestClass() { printf("TestClass - create\n"); }
	TestClass(const TestClass &) { printf("TestClass - copy\n"); }
	TestClass(TestClass &&) { printf("TestClass - move\n"); }
	~TestClass() { printf("TestClass - destroy\n"); }

	void F() { printf("TestClass - F()\n"); }
};

TestClass TestClassCreator()
{
	TestClass c;
	c.F();
	return c;
}

TestClass TestClassProcessor()
{
	TestClass c = TestClassCreator();
	c.F();
	return c;
}

void TestMove()
{
	TestClass c = TestClassProcessor();
}

EXPORT(void) Test()
{
	printf("Hello from C++ DLL\n");

	CreateCC().AddValue(2).AddValue(3);

	printf("Continue working...\n");

	TestMatrix();
	TestMove();

	return;

	ip::Parallel::Do(Test);

	Image<float> img(256, 256);
	img.GetPixelReadIterator(0, 0);
	img.GetVectorWriteIterator(0, 0);


	A a;
	a.Width();
	a.Height();

	B b;
	// b.i2f();
	Test2(b);

	const int n = sizeof(B);
}

EXPORT(void) TestCppUI()
{
	ImageFloat img(256, 16);
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 16; j++)
			img(i, j) = i;

	UI::Display(L"C++ image").AddImage(img, L"Gradient").AddImage(img, L"Gradient2");	

	UI::BeginIterationSet(100000000, L"Iterations", true);

	for (int i = 0; i < 100000000; i++)
	{
		UI::ReportIteration(i);
	}

	UI::EndIterationSet();
}
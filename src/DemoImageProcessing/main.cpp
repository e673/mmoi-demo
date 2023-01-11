#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <warping/meowarping.hpp>
#include <iplib/image/io/imageio.h>
#include <random>
#include <iplib/image/edt/edt.h>
#include <misc/basicedges/basicedges.hpp>
#include <iplib/version.h>
#include <iplib/image/resampling/edresampling.h>
#include <chrono>
#include <resampling/srcnn.h>
#include <resampling/edrfast.h>
#include <resampling/edrvector.h>
#include <resampling/si_resampling.h>
#include <functional>
#include <fstream>
#include <iostream>
#include <iplib/image/deblur/deblurtv.h>

using namespace ip;
using namespace std;

void DisplayHelp()
{
	printf("Program usage:\n");
	printf("(programname) <operation> [options] <input_image> <output_image>\n\n");
	printf("List of operations:\n\n");
	printf("  warp - perform image sharpening by grid warping. Available options:\n");
	printf("    -sigma <value> - (mandatory) set blur parameter for the input image, range: 1.0 to 20.0, default value is 2.0\n");
	printf("    -power <value> - set warping power, range: 0.0 to 5.0, default value is 1.0\n");
	printf("    -1d - apply one-dimensional algorithm (faster but lower quality\n\n");
	printf("  basicedges - detect basic edges - edges good for artifact analysis\n");
	printf("    -scale <value> - set scale parameter, range: 1.0 to 20.0, default value is 4.0\n\n");
	printf("  gaussblur - blur input image using Gauss filter\n");
	printf("    -sigma <value> - (mandatory) Gauss filter radius\n");
	printf("    -noise <value> - Add Gaussian noise with provided stddev\n\n");
	printf("  resample - perform edge-directional resampling (x2)\n");
	printf("    -method <method_name> - Use specific image resampling algorithm:\n");
	printf("      edr (default) - Our edge-directional algorithm\n");
	printf("      srcnn - SRCNN (deep learning)\n");
	printf("      si1, si2, si3 - SI-1, SI-2 and SI-3 respectively\n");
	printf("    -cfile <filename> - Read coefficient data from the specified file\n");
	printf("    -selfsim - Use the input image to compute the interpolation kernels instead of predefined values ('edr' method only)\n\n");
	printf("  train - learn coefficients for edge-directional resampling\n");
	printf("    -in <high_res> <low_res> - use a pair of training images\n");
	printf("    -out <filename> - the result of training (default filename is '(method).bin'");
	printf("    -method <method_name> - one of 'edr' (default), 'si1', 'si2', and 'si3'\n");
	printf("    the rest arguments are filenames of high-resolution training images (low-resolution images are generated)\n\n");

	printf("  help - display this screen\n\n");
	printf("  other operations coming soon...\n\n");
	printf("Formats supported by GdiPlus library can be used: BMP, PNG, JPEG, GIF, TIFF\n");
	exit(0);
}

void Fault(wchar_t *message)
{
	wprintf(L"Error: %s\n", message);
	exit(1);
}

void ProcessWarp(int argc, wchar_t **argv)
{
	wchar_t *input_image = nullptr, *output_image = nullptr, *varsigma_image = nullptr;
	float sigma = -1.0f, power = 1.0f, edge_detection_sigma = -1.0f, varsigma_factor = 0.1f;

	for (int i = 0; i < argc; i++)
	{
		if (lstrcmp(argv[i], L"-varsigma") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -varsigma");

			varsigma_image = argv[i];
		}
		else if (lstrcmp(argv[i], L"-varfactor") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -varfactor");

			size_t idx;
			varsigma_factor = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || varsigma_factor <= 0.0f || varsigma_factor > 1.0f)
				Fault(L"Invalid parameter for -varfactor");
		}
		else if (lstrcmp(argv[i], L"-sigma") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -sigma");

			size_t idx;
			sigma = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || sigma <= 0.0f || sigma > 100.0f)
				Fault(L"Invalid parameter for -sigma");
		}
		else if (lstrcmp(argv[i], L"-power") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -power");

			size_t idx;
			power = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || power <= 0.0f || power > 100.0f)
				Fault(L"Invalid parameter for -power");
		}
		else if (input_image == nullptr)
			input_image = argv[i];
		else if (output_image == nullptr)
			output_image = argv[i];
		else
		{
			wprintf(L"Invalid argument: %s\n", argv[i]);
			exit(1);
		}
	}

	if (sigma < 0.0f)
		Fault(L"Parameter -sigma not defined\n");

	ImageFloatColor4 src = ImageIO::FromFile<PixelFloatRGB4>(input_image);

	MeowWarping mw;
	mw.SetEdgeDetectionSigma(sigma * 0.5f);
	mw.SetWarpingPower(power);

	if (varsigma_image)
	{
		ImageFloat varsigma = ImageIO::FromFile<float>(varsigma_image);

		if (!varsigma)
			Fault(L"Cannot open image");

		if (src.Width() != varsigma.Width() || src.Height() != varsigma.Height())
			Fault(L"Image sigma mismatch");

		for (int j = 0; j < varsigma.Height(); j++)
			for (int i = 0; i < varsigma.Width(); i++)
				varsigma(i, j) *= varsigma_factor;

		mw.SetWarpingSigmaImage(varsigma);
	}
	else
	{
		mw.SetWarpingSigma(sigma);
	}

	ImageFloatColor4 dst = mw.Warp2D(src);

	ImageIO::ToFile(dst, output_image);
}

void ProcessGaussBlur(int argc, wchar_t **argv)
{
	wchar_t *input_image = nullptr, *output_image = nullptr;
	float sigma = -1.0f, noise = -1.0f;

	for (int i = 0; i < argc; i++)
	{
		if (lstrcmp(argv[i], L"-sigma") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -sigma");
			
			size_t idx;
			sigma = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || sigma <= 0.0f || sigma > 100.0f)
				Fault(L"Invalid parameter for -sigma");					
		}
		else if (lstrcmp(argv[i], L"-noise") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -noise");

			size_t idx;
			noise = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || noise <= 0.0f || noise > 10000.0f)
				Fault(L"Invalid parameter for -noise");
		}
		else if (input_image == nullptr)
			input_image = argv[i];
		else if (output_image == nullptr)
			output_image = argv[i];
		else
		{
			wprintf(L"Invalid argument: %s\n", argv[i]);
			exit(1);
		}
	}

	if (sigma < 0.0f)
		Fault(L"Parameter -sigma not defined\n");

	ImageFloatColor src = ImageIO::FromFile<PixelFloatRGBA>(input_image);
	ImageFloatColor res(src.Width(), src.Height());

	GaussFilter(src, res, sigma);

	if (noise > 0.0f)
	{
		default_random_engine random;
		normal_distribution<float> nd(0.0f, noise);

		for (int j = 0; j < res.Height(); j++)
		{
			for (int i = 0; i < res.Width(); i++)
			{
				PixelFloatRGBA p = res(i, j);
				p.r += nd(random);
				p.g += nd(random);
				p.b += nd(random);
				res(i, j) = p;
			}
		}
	}

	ImageIO::ToFile(res, output_image);
}

void ProcessBasicEdges(int argc, wchar_t **argv)
{
	wchar_t *input_image = nullptr, *output_image = nullptr;
	float scale = 4.0f, r1 = 2.0f, r2 = 3.0f;

	for (int i = 0; i < argc; i++)
	{
		if (lstrcmp(argv[i], L"-scale") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -scale");

			size_t idx;
			scale = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || scale <= 0.0f || scale > 100.0f)
				Fault(L"Invalid parameter for -scale");
		}
		else if (lstrcmp(argv[i], L"-r1") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -r1");

			size_t idx;
			r1 = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || r1 < 0.0f || r1 > 100.0f)
				Fault(L"Invalid parameter for -r1");
		}
		else if (lstrcmp(argv[i], L"-r2") == 0)
		{
			i++;
			if (i == argc)
				Fault(L"No parameter for -r2");

			size_t idx;
			r2 = stof(argv[i], &idx);

			if (idx != lstrlen(argv[i]) || r2 <= 0.0f || r2 > 100.0f)
				Fault(L"Invalid parameter for -r2");
		}
		else if (input_image == nullptr)
			input_image = argv[i];
		else if (output_image == nullptr)
			output_image = argv[i];
		else
		{
			wprintf(L"Invalid argument: %s\n", argv[i]);
			exit(1);
		}
	}

	ImageFloatColor src = ImageIO::FromFile<PixelFloatRGBA>(input_image);
	if (!src)
	{
		wprintf(L"Error opening file %s\n", input_image);
		exit(1);
	}

	BasicEdges be = BasicEdges(src.Convert<float>(), scale, r2, r1);
	ImageFloatColor4 res = be.Visualize();

	ImageIO::ToFile(res, output_image);
}

float MeasureExecution(std::function<void()> func)
{
	LARGE_INTEGER pFreq, start, stop;
	QueryPerformanceFrequency(&pFreq);
	QueryPerformanceCounter(&start);
	
	func();

	QueryPerformanceCounter(&stop);

	return (float)(stop.QuadPart - start.QuadPart) / pFreq.QuadPart;
}

ImageByteColor OpenImageByteColor(wchar_t *filename)
{
	ImageByteColor res = ImageIO::FromFile<PixelByteRGBA>(filename);

	if (!res)
	{
		wprintf(L"Error opening file %s\n", filename);
		exit(1);
	}

	return res;
}

std::vector<char> ReadBin(wchar_t *filename)
{
	std::fstream fs;
	fs.open(filename, fstream::in | fstream::binary | fstream::ate);

	if (fs.fail())
	{
		wprintf(L"Error opening file %s: ", filename);
		printf("%s\n", strerror(errno));
		exit(1);
	}

	size_t sz = fs.tellg();
	fs.seekg(0);

	std::vector<char> res(sz);
	fs.read(res.data(), sz);

	fs.close();

	return res;	
}

void ProcessResample(int argc, wchar_t **argv)
{
	wchar_t *input_image = nullptr, *output_image = nullptr, *method = nullptr, *cfile = nullptr;
	bool selfsim = false, half_shift = false;

	for (int i = 0; i < argc; i++)
	{
		if (lstrcmp(argv[i], L"-selfsim") == 0)
		{
			selfsim = true;
		}
		else if (lstrcmp(argv[i], L"-method") == 0)
		{
			++i;
			if (i == argc)
				Fault(L"No parameter for -method");

			method = argv[i];
		}
		else if (lstrcmp(argv[i], L"-cfile") == 0)
		{
			++i;
			if (i == argc)
				Fault(L"No parameter for -method");

			cfile = argv[i];
		}
		else if (lstrcmp(argv[i], L"-halfshift") == 0)
		{
			half_shift = true;
		}
		else if (input_image == nullptr)
		{
			input_image = argv[i];
		}
		else if (output_image == nullptr)
		{
			output_image = argv[i];
		}
		else
		{
			wprintf(L"Invalid argument: %s\n", argv[i]);
			exit(1);
		}
	}

	if (input_image == nullptr)
		Fault(L"No input image specified");

	if (output_image == nullptr)
		Fault(L"No output image specified");

	if (method == nullptr || lstrcmp(method, L"edr") == 0)
	{
		ImageByteColor src = OpenImageByteColor(input_image), dst;
		EDRVector edr;

		if (selfsim)
		{
			printf("Self-similary is not implemented yet\n");
		}

		float time = MeasureExecution([&]()
		{
			dst.swap(edr.Perform(src));
		});

		printf("Resampling EDR %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(), time * 1e3f);

		ImageIO::ToFile(dst, output_image);
	}
	else if (lstrcmp(method, L"srcnn") == 0)
	{
		ImageFloat src = ImageIO::FromFile<float>(input_image);
		if (!src)
		{
			wprintf(L"Error opening file %s\n", input_image);
			exit(1);
		}

		printf("Using SRCNN image resampling algorithm\n");
		printf("The output will be grayscale\n");

		ImageFloat res(src.Width() * 2, src.Height() * 2);

		/* wchar_t buf[2048];
		GetCurrentDirectoryW(2048, buf);
		wprintf(L"%s\n", buf); */

		if (half_shift)
			printf("Half-shift will be used\n");

		SRCNN srcnn(cfile);

		LARGE_INTEGER pFreq, start, stop;
		QueryPerformanceFrequency(&pFreq);
		QueryPerformanceCounter(&start);

		bool s_res = srcnn.Resample_x2_915(src, res, half_shift);

		QueryPerformanceCounter(&stop);

		if (!s_res)
			printf("Error loading SRCNN coefficients\n");

		printf("Resampling SRCNN %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), res.Width(), res.Height(),
			(stop.QuadPart - start.QuadPart) * 1e3f / pFreq.QuadPart);

		ImageIO::ToFile(res, output_image);
	}
	else if (lstrcmp(method, L"si1") == 0)
	{
		ImageByteColor src = OpenImageByteColor(input_image), dst;

		vector<char> bin = ReadBin(cfile != nullptr ? cfile : L"si1.bin");
		SIResampling sir{ SIResampling::Mode::SI1, bin.data(), bin.size() };

		float time = MeasureExecution([&]()
		{
			dst.swap(sir.Perform(src));
		});

		printf("Resampling SI1 %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(), time * 1e3f);

		ImageIO::ToFile(dst, output_image);
	}
	else if (lstrcmp(method, L"si2") == 0)
	{
		ImageByteColor src = OpenImageByteColor(input_image), dst;

		vector<char> bin = ReadBin(cfile != nullptr ? cfile : L"si2.bin");
		SIResampling sir{ SIResampling::Mode::SI2, bin.data(), bin.size() };

		float time = MeasureExecution([&]()
		{
			dst.swap(sir.Perform(src));
		});

		printf("Resampling SI2 %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(), time * 1e3f);

		ImageIO::ToFile(dst, output_image);
	}
	else if (lstrcmp(method, L"si3") == 0)
	{
		ImageByteColor src = OpenImageByteColor(input_image), dst;

		vector<char> bin = ReadBin(cfile != nullptr ? cfile : L"si3.bin");
		SIResampling sir{ SIResampling::Mode::SI3, bin.data(), bin.size() };

		float time = MeasureExecution([&]()
		{
			dst.swap(sir.Perform(src));
		});

		printf("Resampling SI3 %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(), time * 1e3f);

		ImageIO::ToFile(dst, output_image);
	}
	else if (lstrcmp(method, L"si1deblur") == 0)
	{
		ImageByteColor src = OpenImageByteColor(input_image), dst;

		vector<char> bin = ReadBin(cfile != nullptr ? cfile : L"si1deblur.bin");
		SIResampling sir{ SIResampling::Mode::SI1Deblur, bin.data(), bin.size() };

		float time = MeasureExecution([&]()
		{
			dst.swap(sir.PerformDeblur(src));
		});

		printf("Deblurring SI1 %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(), time * 1e3f);

		ImageIO::ToFile(dst, output_image);
	}
	else if (lstrcmp(method, L"si2deblur") == 0)
	{
		ImageByteColor src = OpenImageByteColor(input_image), dst;

		vector<char> bin = ReadBin(cfile != nullptr ? cfile : L"si2deblur.bin");
		SIResampling sir{ SIResampling::Mode::SI2Deblur, bin.data(), bin.size() };

		float time = MeasureExecution([&]()
		{
			dst.swap(sir.PerformDeblur(src));
		});

		printf("Deblurring SI2 %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(), time * 1e3f);

		ImageIO::ToFile(dst, output_image);
	}
	else if (lstrcmp(method, L"si3deblur") == 0)
	{
		ImageByteColor src = OpenImageByteColor(input_image), dst;

		vector<char> bin = ReadBin(cfile != nullptr ? cfile : L"si3deblur.bin");
		SIResampling sir{ SIResampling::Mode::SI3Deblur, bin.data(), bin.size() };

		float time = MeasureExecution([&]()
		{
			dst.swap(sir.PerformDeblur(src));
		});

		printf("Deblurring SI3 %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(), time * 1e3f);

		ImageIO::ToFile(dst, output_image);
	}


	/*



#ifndef LEGACY
	if (method == nullptr || lstrcmp(method, L"edrfast") == 0)
	{
		if (!grayscale)
		{
			ImageFloatColor src = ImageIO::FromFile<PixelFloatRGBA>(input_image);
			if (!src)
			{
				wprintf(L"Error opening file %s\n", input_image);
				exit(1);
			}

			ImageFloatColor dst(src.Width() * 2, src.Height() * 2);

			LARGE_INTEGER pFreq, start, stop;
			QueryPerformanceFrequency(&pFreq);
			QueryPerformanceCounter(&start);

			EDR_Resampling_x2(src, dst);

			QueryPerformanceCounter(&stop);

			printf("Resampling EDR (fast, color) %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(),
				(stop.QuadPart - start.QuadPart) * 1e3f / pFreq.QuadPart);

			ImageIO::ToFile(dst, output_image);
		}
		else
		{
			ImageFloat src = ImageIO::FromFile<float>(input_image);
			if (!src)
			{
				wprintf(L"Error opening file %s\n", input_image);
				exit(1);
			}

			ImageFloat dst(src.Width() * 2, src.Height() * 2);

			LARGE_INTEGER pFreq, start, stop;
			QueryPerformanceFrequency(&pFreq);
			QueryPerformanceCounter(&start);

			EDR_Resampling_x2(src, dst);

			QueryPerformanceCounter(&stop);

			printf("Resampling EDR (fast, gray) %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(),
				(stop.QuadPart - start.QuadPart) * 1e3f / pFreq.QuadPart);

			ImageByte fix(dst.Width(), dst.Height());
			for (int j = 0; j < dst.Height(); j++)
				for (int i = 0; i < dst.Width(); i++)
					fix(i, j) = f2b(dst(i, j));

			ImageIO::ToFile(fix, output_image);
		}
	}
	else if (lstrcmp(method, L"edrvector") == 0)
	{
		{
			ImageByteColor src = ImageIO::FromFile<PixelByteRGBA>(input_image);
			if (!src)
			{
				wprintf(L"Error opening file %s\n", input_image);
				exit(1);
			}

			LARGE_INTEGER pFreq, start, stop;
			QueryPerformanceFrequency(&pFreq);
			QueryPerformanceCounter(&start);

			EDRVector edr;
			ImageByteColor dst = edr.Perform(src);

			QueryPerformanceCounter(&stop);

			printf("Resampling EDR (vector, color) %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), dst.Width(), dst.Height(),
				(stop.QuadPart - start.QuadPart) * 1e3f / pFreq.QuadPart);

			ImageIO::ToFile(dst, output_image);
		}
	}
	else
#endif
		
	if (method == nullptr || lstrcmp(method, L"edr") == 0)
	{
		ImageFloatColor src = ImageIO::FromFile<PixelFloatRGBA>(input_image);
		if (!src)
		{
			wprintf(L"Error opening file %s\n", input_image);
			exit(1);
		}

		printf("Using EDR image resampling algorithm\n");

		if (selfsim)
			wprintf(L"Using the input image as the source of kernel coefficients\n");
		else
			wprintf(L"Using the default coefficients\n");

		if (fast)
			wprintf(L"Fast weight calculation is turned on\n");

		ImageFloatColor res(src.Width() * 2, src.Height() * 2);

		EDResamplingExt edr;
		edr.Configure(fast);

		if (selfsim)
		{
			ImageFloatColor tmp(src.Width(), src.Height());
			ImageFloatColor lr(src.Width() / 2, src.Height() / 2);

			GaussFilter(src, tmp, 0.7f);
			for (int j = 0; j < lr.Height(); j++)
				for (int i = 0; i < lr.Width(); i++)
					lr(i, j) = tmp(i * 2, j * 2);

			edr.LearnStep1(lr, src);
			edr.UpdateCoefficientsStep1();
			edr.LearnStep2(lr, src);
			edr.UpdateCoefficientsStep2();
		}

		LARGE_INTEGER pFreq, start, stop;
		QueryPerformanceFrequency(&pFreq);
		QueryPerformanceCounter(&start);

		edr.Perform(src, res, EDResampling::Mode::Isotropic);

		QueryPerformanceCounter(&stop);
		printf("Resampling EDR (color) %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), res.Width(), res.Height(),
			(stop.QuadPart - start.QuadPart) * 1e3f / pFreq.QuadPart);
		ImageIO::ToFile(res, output_image);
	}
	else if (lstrcmp(method, L"srcnn") == 0)
	{
		ImageFloat src = ImageIO::FromFile<float>(input_image);
		if (!src)
		{
			wprintf(L"Error opening file %s\n", input_image);
			exit(1);
		}

		printf("Using SRCNN image resampling algorithm\n");
		printf("The output will be grayscale\n");

		ImageFloat res(src.Width() * 2, src.Height() * 2);

		LARGE_INTEGER pFreq, start, stop;
		QueryPerformanceFrequency(&pFreq);
		QueryPerformanceCounter(&start);

		SRCNN_Resampling_x2(src, res);

		QueryPerformanceCounter(&stop);
		printf("Resampling SRCNN %dx%d -> %dx%d execution time: %.3f ms\n", src.Width(), src.Height(), res.Width(), res.Height(),
			(stop.QuadPart - start.QuadPart) * 1e3f / pFreq.QuadPart);

		ImageIO::ToFile(res, output_image);
	}
	else if (lstrcmp(method, L"si1") == 0)
	{
		printf("Sorry, we plan to implement the algorithm in C++ but we are too lazy\n");
	}
	else
	{
		wprintf(L"Unknown method - %s\n", method);
	} */
}

class TP
{
public:
	TP() { printf("Default constructed\n"); }
	TP(const TP&) { printf("Copy constructed\n"); }
	~TP() { printf("Destructed\n"); }

	void Test()
	{

	}
};

template <class T>
void print_matrix(int x, int y, T &oper)
{
	for (int j = 0; j < y; j++)
	{
		oper(0, j);
		for (int i = 1; i < x; i++)
		{
			printf("\t");
			oper(i, j);
		}
		printf("\n");
	}
	printf("\n");
}

void TestEDT()
{
	ImageBinary src(8, 8);
	Image<int> res1(src.Width(), src.Height());
	Image<int> res2(src.Width(), src.Height());
	Image<int> res3(src.Width(), src.Height());
	Image<PixelVector<int>> ref(src.Width(), src.Height());

	default_random_engine random;
	uniform_int_distribution<> distr(0, 1);

	for (int iter = 0; iter < 100000; iter++)
	{
		int cnt = 0;

		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				if (distr(random) == 1)
				{
					src(i, j) = true;
					cnt++;
				}
				else
					src(i, j) = false;

		if (cnt == 0)
			continue;

		EDT::Slow(src, res1);
		EDT::Simple(src, res2);
		EDT::Extended(src, res3, ref);

		bool ok = true;

		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				if (res1(i, j) != res2(i, j))
					ok = false;

				if (res1(i, j) != res3(i, j))
					ok = false;

				auto v = ref(i, j);
				if ((v.x - i) * (v.x - i) + (v.y - j) * (v.y - j) != res1(i, j))
					ok = false;
			}
		}

		if (ok)
			continue;

		printf("EDT ERROR:\n");

		print_matrix(src.Width(), src.Height(), [&src](int x, int y) { printf("%d", src(x, y) ? 1 : 0); });
		print_matrix(src.Width(), src.Height(), [&res1](int x, int y) { printf("%d", res1(x, y)); });
		print_matrix(src.Width(), src.Height(), [&res2](int x, int y) { printf("%d", res2(x, y)); });
		print_matrix(src.Width(), src.Height(), [&res3, &ref](int x, int y) { printf("%d (%d, %d)", res3(x, y), ref(x, y).x, ref(x, y).y ); });

		EDT::Extended(src, res3, ref);
	}
}

void LearnSIResampling(SIResampling &sir, const std::vector<wchar_t*> hr_files, const std::vector<wchar_t*> lr_files)
{
	for (size_t i = 0; i < hr_files.size(); i++)
	{
		wchar_t *hr_file = hr_files[i];
		wchar_t *lr_file = lr_files[i];

		ImageByteColor hr = ImageIO::FromFile<PixelByteRGBA>(hr_file);
		if (!hr)
		{
			wprintf(L"Cannot open file %s\n", hr_file);
			exit(1);
		}

		if (lr_file == nullptr)
		{
			ImageByteColor lr = ImageByteColor(hr.Width() / 2, hr.Height() / 2);

			Parallel::For(0, lr.Height(), [&lr, &hr](int j)
			{
				for (int i = 0; i < lr.Width(); i++)
				{
					PixelFloatRGBA v0 = hr(2 * i, 2 * j);
					PixelFloatRGBA v1 = hr(2 * i + 1, 2 * j);
					PixelFloatRGBA v2 = hr(2 * i, 2 * j + 1);
					PixelFloatRGBA v3 = hr(2 * i + 1, 2 * j + 1);

					PixelFloatRGBA vr = (v0 + v1 + v2 + v3) * 0.25f;

					lr(i, j) = (PixelByteRGB)vr;
				}
			});

			wprintf(L"Learning coefficients from %s\n", hr_file);

			sir.AddLearningImage(lr, hr);
		}
		else
		{
			ImageByteColor lr = ImageIO::FromFile<PixelByteRGBA>(lr_file);
			if (!lr)
			{
				wprintf(L"Cannot open file %s\n", lr_file);
				exit(1);
			}

			wprintf(L"Learning coefficients from pair (%s, %s)\n", hr_file, lr_file);

			sir.AddLearningImage(lr, hr);
		}
	}

	int min, aver, max;
	sir.GetLearningStats(min, aver, max);

	printf("Learning stats per class: min = %d, aver = %d, max = %d\n", min, aver, max);
}

void LearnSIDeblur(SIResampling &sir, const std::vector<wchar_t*> hr_files, const std::vector<wchar_t*> lr_files)
{
	for (size_t i = 0; i < hr_files.size(); i++)
	{
		wchar_t *hr_file = hr_files[i];
		wchar_t *lr_file = lr_files[i];

		ImageByteColor hr = ImageIO::FromFile<PixelByteRGBA>(hr_file);
		if (!hr)
		{
			wprintf(L"Cannot open file %s\n", hr_file);
			exit(1);
		}

		if (lr_file == nullptr)
		{
			ImageByteColor lr = ImageByteColor(hr.Width(), hr.Height());

			GaussFilter(hr, lr, 1.0f);

			wprintf(L"Learning coefficients from %s\n", hr_file);

			sir.AddLearningImageDeblur(lr, hr);
		}
		else
		{
			ImageByteColor lr = ImageIO::FromFile<PixelByteRGBA>(lr_file);
			if (!lr)
			{
				wprintf(L"Cannot open file %s\n", lr_file);
				exit(1);
			}

			wprintf(L"Learning coefficients from pair (%s, %s)\n", hr_file, lr_file);

			sir.AddLearningImageDeblur(lr, hr);
		}
	}

	int min, aver, max;
	sir.GetLearningStats(min, aver, max);

	printf("Learning stats per class: min = %d, aver = %d, max = %d\n", min, aver, max);
}

void StoreSICoefficients(SIResampling &sir, const wchar_t *filename)
{
	sir.PerformLearning();

	size_t size = sir.SaveCoefficientData(nullptr, 0);
	std::unique_ptr<char[]> data(new char[size]);
	sir.SaveCoefficientData(data.get(), size);

	FILE *F = _wfopen(filename, L"wb");
	fwrite(data.get(), 1, size, F);
	fclose(F);
}

void ProcessTrain(int argc, wchar_t **argv)
{
	wchar_t *method = nullptr, *out_filename = nullptr;
	std::vector<wchar_t*> hr_files;
	std::vector<wchar_t*> lr_files;

	for (int i = 0; i < argc; i++)
	{
		if (lstrcmp(argv[i], L"-method") == 0)
		{
			++i;
			if (i == argc)
				Fault(L"Missing argument for -method");
			else
				method = argv[i];
		}
		else if (lstrcmp(argv[i], L"-in") == 0)
		{
			i += 2;
			if (i == argc)
			{
				Fault(L"Missing arguments for -in");
			}
			else
			{
				hr_files.push_back(argv[i - 1]);
				lr_files.push_back(argv[i]);
			}
		}
		else if (lstrcmp(argv[i], L"-out") == 0)
		{
			++i;
			if (i == argc)
				Fault(L"Missing argument for -out");
			else
				out_filename = argv[i];
		}
		else
		{
			hr_files.push_back(argv[i]);
			lr_files.push_back(nullptr);
		}
	}

	if (hr_files.size() == 0)
		Fault(L"No training images provided");

	if ((method == nullptr) || (lstrcmp(method, L"edr") == 0))
	{

	}
	else if (lstrcmp(method, L"si1") == 0)
	{
		SIResampling sir(SIResampling::Mode::SI1);
		LearnSIResampling(sir, hr_files, lr_files);
		StoreSICoefficients(sir, out_filename != nullptr ? out_filename : L"si1.bin");
	}
	else if (lstrcmp(method, L"si2") == 0)
	{
		SIResampling sir(SIResampling::Mode::SI2);
		LearnSIResampling(sir, hr_files, lr_files);
		StoreSICoefficients(sir, out_filename != nullptr ? out_filename : L"si2.bin");
	}
	else if (lstrcmp(method, L"si3") == 0)
	{
		SIResampling sir(SIResampling::Mode::SI3);
		LearnSIResampling(sir, hr_files, lr_files);
		StoreSICoefficients(sir, out_filename != nullptr ? out_filename : L"si3.bin");
	}
	else if (lstrcmp(method, L"si1deblur") == 0)
	{
		SIResampling sir(SIResampling::Mode::SI1Deblur);
		LearnSIDeblur(sir, hr_files, lr_files);
		StoreSICoefficients(sir, out_filename != nullptr ? out_filename : L"si1deblur.bin");
	}
	else if (lstrcmp(method, L"si2deblur") == 0)
	{
		SIResampling sir(SIResampling::Mode::SI2Deblur);
		LearnSIDeblur(sir, hr_files, lr_files);
		StoreSICoefficients(sir, out_filename != nullptr ? out_filename : L"si2deblur.bin");
	}
	else if (lstrcmp(method, L"si3deblur") == 0)
	{
		SIResampling sir(SIResampling::Mode::SI3Deblur);
		LearnSIDeblur(sir, hr_files, lr_files);
		StoreSICoefficients(sir, out_filename != nullptr ? out_filename : L"si3deblur.bin");
	}
	else
	{
		Fault(L"Unsupported method");
	}
}

void PerfTest()
{
	const int N = 512;

	Image<float> img1f(N, N), img2f(N, N);
	Image<byte> img1b(N, N), img2b(N, N);

	for (int j = 0; j < img1f.Height(); j++)
		for (int i = 0; i < img1f.Width(); i++)
		{
			img1f(i, j) = img2f(i, j) = 3 * i + j;
			img1b(i, j) = img2b(i, j) = 3 * i + j;
		}

	long long freq = _Query_perf_frequency();
	long long t0 = _Query_perf_counter();

	for (int j = 1; j < img1f.Height() - 1; j++)
	{
		float *l0 = img1f.pixeladdr(1, j - 1);
		float *l1 = img1f.pixeladdr(1, j);
		float *l2 = img1f.pixeladdr(1, j + 1);
		float *dst = img2f.pixeladdr(1, j);

		for (int i = 1; i < img1f.Width(); i++)
		{
			float q1 = l0[-1] + l0[1] + l2[-1] + l2[1];
			float q2 = l0[0] + l1[-1] + l1[1] + l2[0];
			float q4 = l1[0];

			*dst = (q1 + 2.0f * q2 + 4.0f * q4) * (1.0f / 16.0f);

			l0++;
			l1++;
			l2++;
			dst++;
		}
	}

	check(img2f(10, 10) == 40.0f);

	long long t1 = _Query_perf_counter();

	for (int j = 1; j < img1b.Height() - 1; j++)
	{
		byte *l0 = img1b.pixeladdr(1, j - 1);
		byte *l1 = img1b.pixeladdr(1, j);
		byte *l2 = img1b.pixeladdr(1, j + 1);
		byte *dst = img2b.pixeladdr(1, j);

		for (int i = 1; i < img1f.Width(); i++)
		{
			int q1 = l0[-1] + l0[1] + l2[-1] + l2[1];
			int q2 = l0[0] + l1[-1] + l1[1] + l2[0];
			int q4 = l1[0];

			*dst = (q1 + 2 * q2 + 4 * q4) >> 4;

			l0++;
			l1++;
			l2++;
			dst++;
		}
	}

	check(img2b(10, 10) == 40);

	long long t2 = _Query_perf_counter();

	const __m128 m16 = _mm_set1_ps(1.0f / 16.0f);

	for (int j = 1; j < img1f.Height() - 1; j++)
	{
		float *l0 = img1f.pixeladdr(1, j - 1);
		float *l1 = img1f.pixeladdr(1, j);
		float *l2 = img1f.pixeladdr(1, j + 1);
		float *dst = img2f.pixeladdr(1, j);

		// We do not care about boundary processing. Just benchmark.
		for (int i = 1; i < img1f.Width() - 4; i += 4)
		{
			__m128 q1 = _mm_add_ps(_mm_add_ps(_mm_loadu_ps(l0 - 1), _mm_loadu_ps(l0 + 1)), _mm_add_ps(_mm_loadu_ps(l2 - 1), _mm_loadu_ps(l2 + 1)));
			__m128 q2 = _mm_add_ps(_mm_add_ps(_mm_loadu_ps(l0), _mm_loadu_ps(l2)), _mm_add_ps(_mm_loadu_ps(l1 - 1), _mm_loadu_ps(l1 + 1)));
			__m128 q4 = _mm_loadu_ps(l1);
			q2 = _mm_add_ps(q2, q2);
			q4 = _mm_add_ps(q4, q4);
			__m128 res = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(q4, q4), q2), q1), m16);
			_mm_storeu_ps(dst, res);

			l0 += 4;
			l1 += 4;
			l2 += 4;
			dst += 4;
		}
	}

	check(img2f(10, 10) == 40.0f);

	long long t3 = _Query_perf_counter();

	const __m256 q16 = _mm256_set1_ps(1.0f / 16.0f);

	for (int j = 1; j < img1f.Height() - 1; j++)
	{
		float *l0 = img1f.pixeladdr(1, j - 1);
		float *l1 = img1f.pixeladdr(1, j);
		float *l2 = img1f.pixeladdr(1, j + 1);
		float *dst = img2f.pixeladdr(1, j);

		// We do not care about boundary processing. Just benchmark.
		for (int i = 1; i < img1f.Width() - 8; i += 8)
		{
			__m256 q1 = _mm256_add_ps(_mm256_add_ps(_mm256_load_ps(l0 - 1), _mm256_load_ps(l0 + 1)), _mm256_add_ps(_mm256_load_ps(l2 - 1), _mm256_load_ps(l2 + 1)));
			__m256 q2 = _mm256_add_ps(_mm256_add_ps(_mm256_load_ps(l0), _mm256_load_ps(l2)), _mm256_add_ps(_mm256_load_ps(l1 - 1), _mm256_load_ps(l1 + 1)));
			__m256 q4 = _mm256_load_ps(l1);
			q2 = _mm256_add_ps(q2, q2);
			q4 = _mm256_add_ps(q4, q4);
			__m256 res = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(q4, q4), q2), q1), q16);
			_mm256_storeu_ps(dst, res);

			l0 += 8;
			l1 += 8;
			l2 += 8;
			dst += 8;
		}
	}

	check(img2f(10, 10) == 40.0f);

	long long t4 = _Query_perf_counter();

	for (int j = 1; j < img1f.Height() - 1; j++)
	{
		float *l0 = img1f.pixeladdr(1, j - 1);
		float *l1 = img1f.pixeladdr(1, j);
		float *l2 = img1f.pixeladdr(1, j + 1);
		float *dst = img2f.pixeladdr(1, j);

		// We do not care about boundary processing. Just benchmark.
		for (int i = 1; i < img1f.Width() - 8; i += 8)
		{
			__m256 q1 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(l0 - 1), _mm256_loadu_ps(l0 + 1)), _mm256_add_ps(_mm256_loadu_ps(l2 - 1), _mm256_loadu_ps(l2 + 1)));
			__m256 q2 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(l0), _mm256_loadu_ps(l2)), _mm256_add_ps(_mm256_loadu_ps(l1 - 1), _mm256_loadu_ps(l1 + 1)));
			__m256 q4 = _mm256_loadu_ps(l1);
			q2 = _mm256_add_ps(q2, q2);
			q4 = _mm256_add_ps(q4, q4);
			__m256 res = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(q4, q4), q2), q1), q16);
			_mm256_storeu_ps(dst, res);

			l0 += 8;
			l1 += 8;
			l2 += 8;
			dst += 8;
		}
	}

	check(img2f(10, 10) == 40.0f);

	long long t5 = _Query_perf_counter();

	for (int j = 1; j < img1b.Height() - 1; j++)
	{
		byte *l0 = img1b.pixeladdr(1, j - 1);
		byte *l1 = img1b.pixeladdr(1, j);
		byte *l2 = img1b.pixeladdr(1, j + 1);
		byte *dst = img2b.pixeladdr(1, j);

		for (int i = 1; i < img1f.Width() - 8; i += 8)
		{
			__m128i v1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l0 - 1)), _mm_setzero_si128());
			__m128i v2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l0 + 1)), _mm_setzero_si128());
			__m128i v3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l2 - 1)), _mm_setzero_si128());
			__m128i v4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l2 + 1)), _mm_setzero_si128());
			__m128i q1 = _mm_add_epi16(_mm_add_epi16(v1, v2), _mm_add_epi16(v3, v4));

			__m128i v5 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l0)), _mm_setzero_si128());
			__m128i v6 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l2)), _mm_setzero_si128());
			__m128i v7 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l1 - 1)), _mm_setzero_si128());
			__m128i v8 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l1 + 1)), _mm_setzero_si128());
			__m128i q2 = _mm_add_epi16(_mm_add_epi16(v5, v6), _mm_add_epi16(v7, v8));

			__m128i q4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(l1)), _mm_setzero_si128());

			q2 = _mm_add_epi16(q2, q2);
			q4 = _mm_add_epi16(q4, q4);
			__m128i res = _mm_packus_epi16(_mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(q4, q4), q2), q1), 4), _mm_setzero_si128());

			_mm_storel_epi64((__m128i*)dst, res);

			l0 += 8;
			l1 += 8;
			l2 += 8;
			dst += 8;
		}
	}

	check(img2b(10, 10) == 40);

	long long t6 = _Query_perf_counter();

	printf("float: %lld us\n", (t1 - t0) * 1000000 / freq);
	printf("byte: %lld us\n", (t2 - t1) * 1000000 / freq);
	printf("float sse: %lld us\n", (t3 - t2) * 1000000 / freq);
	printf("float avx (aligned): %lld us\n", (t4 - t3) * 1000000 / freq);
	printf("float avx (unaligned): %lld us\n", (t5 - t4) * 1000000 / freq);
	printf("byte sse: %lld us\n", (t6 - t5) * 1000000 / freq);

	getchar();
}

void ProcessGTV(int argc, wchar_t **argv)
{
	if (argc != 5)
	{
		printf("gtv (in_file) (in_kernel) (out_file) (alpha1) (alpha2)\n");
		return;
	}

	ImageFloat src = ImageIO::FromFile<float>(argv[0]);
	ImageFloat kernel = ImageIO::FromFile<float>(argv[1]);
	ImageFloat dst(src.Width(), src.Height());

	float a1 = (float)_wtof(argv[3]);
	float a2 = (float)_wtof(argv[4]);

	float s = 0.0f;

	for (int j = 0; j < kernel.Height(); j++)
		for (int i = 0; i < kernel.Width(); i++)
			s += kernel(i, j);

	for (int j = 0; j < kernel.Height(); j++)
		for (int i = 0; i < kernel.Width(); i++)
			kernel(i, j) /= s;

	DeblurTV::AnyKernel2(src, dst, kernel, a1, a2, 50, 0.8f, 1.0f);

	for (int j = 0; j < dst.Height(); j++)
		for (int i = 0; i < dst.Width(); i++)
			dst(i, j) = f2b(dst(i, j));

	ImageIO::ToFile(dst, argv[2]);
}

int wmain(int argc, wchar_t **argv)
{
	// PerfTest();

	printf("DemoImageProcessing, build %s\n", ip::CompileDateTime);
	printf("(c) Laboratory of Mathematical Methods of Image Processing\n");
	printf("Faculty of Computational Mathematics and Cybernetics\n");
	printf("Lomonosov Moscow State University\n");		
	printf("http://imaging.cs.msu.ru/ Contact nasonov@cs.msu.ru\n\n");

	if (argc <= 1)
		DisplayHelp();

	if (lstrcmp(argv[1], L"warp") == 0)
		ProcessWarp(argc - 2, argv + 2);
	else if (lstrcmp(argv[1], L"gaussblur") == 0)
		ProcessGaussBlur(argc - 2, argv + 2);
	else if (lstrcmp(argv[1], L"basicedges") == 0)
		ProcessBasicEdges(argc - 2, argv + 2);
	else if (lstrcmp(argv[1], L"resample") == 0)
		ProcessResample(argc - 2, argv + 2);
	else if (lstrcmp(argv[1], L"train") == 0)
		ProcessTrain(argc - 2, argv + 2);
	else if (lstrcmp(argv[1], L"gtv") == 0)
		ProcessGTV(argc - 2, argv + 2);
	else
		wprintf(L"Unknown operation - %s\n", argv[1]);

	// Don't forget to cleanup resources
	Parallel::Reset();

#ifndef FULLRELEASE

	getchar();

#endif


	return 0;
};
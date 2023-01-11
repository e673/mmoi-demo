#pragma once

#include "../core.h"
#include <iplib/parallel.h>

namespace ip
{
	// Note: this class is NOT thread-safe
	class StandardEDT
	{
	public:
		StandardEDT(int N);

		void Process(int *src);
		void ProcessFirst(int *src);

		int* GetInput();
		int* GetResultDistance();
		int* GetResultPositions();

	private:
		int *src;
		std::unique_ptr<int[]> input, dist, prev, next, ref;
		int N, first, last;

		void Init(int *src);
		void ConstructLists();
		void NonUsedPointsRemoval();
		void WriteResult();
	};

	// ==================================================================================================

	class EDT
	{
	public:
		static void OneDimensional(int *src, int *dst, int N);
		static void OneDimensional(bool *src, int *dst, int N);

		template <class SourceImageType, class DestinationImageType>
		static void Slow(const ImageReadable<bool, SourceImageType> &src, ImageWritable<int, DestinationImageType> &dst);

		template <class SourceImageType, class DestinationImageType>
		static void Simple(const ImageReadable<bool, SourceImageType> &src, ImageWritable<int, DestinationImageType> &dst);

		template <class SourceImageType, class DestinationImageType, class VectorImageType>
		static void Extended(const ImageReadable<bool, SourceImageType> &src, ImageWritable<int, DestinationImageType> &dst,
			ImageWritable<PixelVector<int>, VectorImageType> &vec);

	};

	// ==================================================================================================

	template <class SourceImageType, class DestinationImageType>
	void EDT::Slow(const ImageReadable<bool, SourceImageType> &src, ImageWritable<int, DestinationImageType> &dst)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());

		Parallel::For(0, src.Height(), [&src, &dst](int j)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				int best = -1;

				for (int y = 0; y < src.Height(); y++)
				{
					for (int x = 0; x < src.Width(); x++)
					{
						if (!src(x, y))
							continue;

						int d = (x - i) * (x - i) + (y - j) * (y - j);
						if (d < best || best == -1)
							best = d;
					}
				}

				dst(i, j) = best;
			}
		});
	}


	template <class SourceImageType, class DestinationImageType>
	void EDT::Simple(const ImageReadable<bool, SourceImageType> &src, ImageWritable<int, DestinationImageType> &dst)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());

		Image<int> tmp(src.Height(), src.Width());
		StandardEDT step1(src.Width());

		// Parallel::For(0, src.Height(), [step1, &src, &tmp](int j) mutable
		for (int j = 0; j < src.Height(); j++)
		{
			int* input = step1.GetInput();
			for (int i = 0; i < src.Width(); i++)
				input[i] = src(i, j) ? 0 : -1;

			step1.ProcessFirst(input);

			for (int i = 0; i < src.Width(); i++)
				tmp(j, i) = step1.GetResultDistance()[i];
		} //);

		StandardEDT step2(src.Height());

		Parallel::For([&tmp, &dst](std::atomic_int &cnt)
		{
			StandardEDT step2(dst.Height());
			for (int i = cnt++; i < dst.Width(); i = cnt++)
			{
				step2.Process(tmp.pixeladdr(0, i));
				for (int j = 0; j < dst.Height(); j++)
					dst(i, j) = step2.GetResultDistance()[j];
			}
		});
	}

	template <class SourceImageType, class DestinationImageType, class VectorImageType>
	void EDT::Extended(const ImageReadable<bool, SourceImageType> &src, ImageWritable<int, DestinationImageType> &dst,
		ImageWritable<PixelVector<int>, VectorImageType> &vec)
	{
		check(src.Width() == dst.Width() && src.Height() == dst.Height());

		Image<int> tmp(src.Height(), src.Width());
		Image<int> vref(src.Height(), src.Width());

		Parallel::For([&src, &tmp, &vref](std::atomic_int &cnt)
		{
			StandardEDT step1(src.Width());

			for (int j = cnt++; j < src.Height(); j = cnt++)
			{
				int* input = step1.GetInput();
				for (int i = 0; i < src.Width(); i++)
					input[i] = src(i, j) ? 0 : -1;

				step1.ProcessFirst(input);

				for (int i = 0; i < src.Width(); i++)
				{
					tmp(j, i) = step1.GetResultDistance()[i];
					vref(j, i) = step1.GetResultPositions()[i];
					// vec(i, j) = PixelVector<int>(step1.GetResultPositions()[i], j);
				}
			}
		});

		Parallel::For([&tmp, &dst, &vref, &vec](std::atomic_int &cnt)
		{
			StandardEDT step2(dst.Height());

			for (int i = cnt++; i < dst.Width(); i = cnt++)
			{
				step2.Process(tmp.pixeladdr(0, i));
				for (int j = 0; j < dst.Height(); j++)
				{
					dst(i, j) = step2.GetResultDistance()[j];
					int p = step2.GetResultPositions()[j];
					vec(i, j) = PixelVector<int>(vref(p, i), p);
				}
			}
		});
	}



}
#pragma once

#include "../../iplib/image/edt/edt.h"

namespace ip
{
	// ==================================================================================================

	StandardEDT::StandardEDT(int N)
		: N(N), src(nullptr), dist(new int[N]), ref(new int[N]), prev(new int[N]), next(new int[N])
	{}

	void StandardEDT::Init(int *src)
	{
		this->src = src;

		first = 0;
		last = N - 1;

		while (src[first] == -1 && first <= last)
			first++;

		while (src[last] == -1 && first <= last)
			last--;
	}

	void StandardEDT::ConstructLists()
	{
		int v = -1;

		for (int i = first; i <= last; i++)
		{
			if (src[i] != -1)
			{
				prev[i] = v;
				v = i;
			}
		}

		v = -1;
		for (int i = last; i >= first; i--)
		{
			if (src[i] != -1)
			{
				next[i] = v;
				v = i;
			}
		}
	}

	void StandardEDT::NonUsedPointsRemoval()
	{
		if (first < last)
		{
			int x2 = next[first];

			while (x2 < last)
			{
				int x1 = prev[x2], x3 = next[x2];
				int y1 = src[x1], y2 = src[x2], y3 = src[x3];
				int z1 = x1 * x1 + y1, z2 = x2 * x2 + y2, z3 = x3 * x3 + y3;

				if (z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2) > 0)
				{
					next[x1] = x3;
					prev[x3] = x1;

					x2 = (x1 == first) ? x3 : x1;
				}
				else
				{
					x2 = x3;
				}
			}
		}
	}

	void StandardEDT::WriteResult()
	{
		if (first > last)
		{
			for (int i = 0; i < N; i++)
			{
				ref[i] = -1;
				dist[i] = -1;
			}

			return;
		}

		int cur = first;

		for (int i = 0; i < N; i++)
		{
			int D = (cur - i) * (cur - i) + src[cur];

			while (cur < last)
			{
				int ncur = next[cur];
				int D2 = (ncur - i) * (ncur - i) + src[ncur];

				if (D < D2)
				{
					break;
				}
				else
				{
					cur = next[cur];
					D = D2;
				}
			}

			ref[i] = cur;
			dist[i] = D;
		}
	}

	void StandardEDT::Process(int *src)
	{
		Init(src);
		ConstructLists();
		NonUsedPointsRemoval();
		WriteResult();
	}

	void StandardEDT::ProcessFirst(int *src)
	{
		int first = 0, last = N - 1;

		// Find first TRUE element
		while (first < N && src[first] != 0)
			first++;

		// No TRUE elements => special output
		if (first == N)
		{
			for (int i = 0; i < N; i++)
				dist[i] = ref[i] = -1;
			return;
		}

		// Find last TRUE element
		while (last >= 0 && src[last] != 0)
			last--;

		int j;

		// Fill first FALSE elements
		for (j = 0; j < first; j++)
		{
			dist[j] = (first - j) * (first - j);
			ref[j] = first;
		}

		// Fill first TRUE element
		dist[j] = 0;
		ref[j] = j;

		// Fill all remaining elements - forward
		for (int i = j + 1; i < N; i++)
		{
			if (src[i] == 0)
			{
				j = i;
				dist[i] = 0;
			}
			else
			{
				dist[i] = (i - j) * (i - j);
			}

			ref[i] = j;
		}

		// Fill all remaining elements - backward
		j = last;
		for (int i = last - 1; i > first; i--)
		{
			if (src[i] == 0)
			{
				j = i;
			}
			else
			{
				int v = (i - j) * (i - j);
				if (v < dist[i])
				{
					dist[i] = v;
					ref[i] = j;
				}
			}
		}
	}

	int* StandardEDT::GetInput()
	{
		if (!input)
			input.reset(new int[N]);

		return input.get();
	}

	int* StandardEDT::GetResultDistance()
	{
		return dist.get();
	}

	int* StandardEDT::GetResultPositions()
	{
		return ref.get();
	}


	// ==================================================================================================

	void EDT::OneDimensional(int *src, int *dst, int N)
	{
		StandardEDT edt(N);

		edt.Process(src);

		for (int i = 0; i < N; i++)
			dst[i] = edt.GetResultDistance()[i];
	}

	void EDT::OneDimensional(bool *src, int *dst, int N)
	{
		StandardEDT edt(N);
		int *input = edt.GetInput();
		for (int i = 0; i < N; i++)
			input[i] = src[i] ? 0 : -1;

		edt.ProcessFirst(input);

		for (int i = 0; i < N; i++)
			dst[i] = edt.GetResultDistance()[i];
	}
}
#pragma once

#include "../core.h"

namespace ip
{
	class Diffusion
	{
	public:
		static void PeronaMalikDivisionIteration(const ip::ImageFloat& src, ip::ImageFloat& dst, float dt, float k);
		static void PeronaMalikExponentIteration(const ip::ImageFloat& src, ip::ImageFloat& dst, float dt, float k);

	private:
		const ip::ImageFloat& src;
		ip::ImageFloat& dst;
		float dt, k;

		Diffusion(const ip::ImageFloat& src, ip::ImageFloat& dst, float dt, float k);

		void IteratePeronaMalikDivisionSafe(int x, int y);
		void IteratePeronaMalikExponentSafe(int x, int y);

		float GetSafe(int x, int y);
		float CalcGradientSafe(int x, int y);
		float CalcLaplasSafe(int x, int y);
	};
}
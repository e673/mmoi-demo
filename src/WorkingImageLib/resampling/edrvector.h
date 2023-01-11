#pragma once

#include <misc/vectorimage.h>

namespace ip
{
	// Required padding: 3 pixels left/top amd 4 pixels right/bottom
	class EDRVector
	{
		class Impl;

		Impl *impl;

	public:
		EDRVector();
		~EDRVector();

		ImageByteColor Perform(const ImageByteColor &lr);

		void Perform(const VectorImageFloatColor &padded_lr, Image<VectorByte> &dst);
		
		void LearnStep1(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr);
		void UpdateCoefficientsStep1();

		void LearnStep2(const VectorImageFloatColor &padded_lr, VectorImageFloatColor &hr);
		void UpdateCoefficientsStep2();
	};
}
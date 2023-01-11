#pragma once

#include <iplib/image/core.h>
#include <misc/vectorimage.h>

namespace ip
{
	class SIResampling
	{
		class Impl;
		
		Impl *impl;

	public:

		enum class Mode	{ SI1, SI2, SI3, SI1Deblur, SI2Deblur, SI3Deblur };

		SIResampling(Mode mode);
		SIResampling(Mode mode, void *coefficient_data, size_t coefficient_data_size);
		~SIResampling();

		ImageByteColor Perform(const ImageByteColor &lr);
		ImageByteColor PerformDeblur(const ImageByteColor &lr);

		void AddLearningImage(const ImageByteColor &lr, const ImageByteColor &hr);
		void AddLearningImageDeblur(const ImageByteColor &lr, const ImageByteColor &hr);
		void PerformLearning();
		void GetLearningStats(int &min, int &aver, int &max);

		size_t SaveCoefficientData(void *buffer, size_t buffer_length);
	};
}
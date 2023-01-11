#pragma once

#include <iplib/image/core.h>
#include "simd.h"

namespace ip
{

	struct VectorFloatColor
	{
		VectorFloat y;
		VectorFloat u;
		VectorFloat v;

		inline VectorFloatColor() = default;
		inline VectorFloatColor(VectorFloat y, VectorFloat u, VectorFloat v);

		inline VectorFloatColor operator + (const VectorFloatColor &right) const;
		inline VectorFloatColor operator - (const VectorFloatColor &right) const;
		inline VectorFloatColor operator * (const VectorFloat &right) const;
		inline VectorFloatColor operator * (float right) const;

		inline VectorByte ToByte() const;
	};

	struct VectorFloatColorA
		: public VectorFloatColor
	{
		VectorFloat a;
	};

	typedef Image<VectorFloat> VectorImageFloat;
	typedef Image<VectorFloatColor> VectorImageFloatColor;

	// ==================================================================================================

	inline VectorFloatColor::VectorFloatColor(VectorFloat y, VectorFloat u, VectorFloat v)
		: y(y), u(u), v(v) {}

	inline VectorFloatColor VectorFloatColor::operator + (const VectorFloatColor &right) const
	{
		return VectorFloatColor(y + right.y, u + right.u, v + right.v);
	}

	inline VectorFloatColor VectorFloatColor::operator - (const VectorFloatColor &right) const
	{
		return VectorFloatColor(y - right.y, u - right.u, v - right.v);
	}

	inline VectorFloatColor VectorFloatColor::operator * (const VectorFloat &right) const
	{
		return VectorFloatColor(y * right, u * right, v * right);
	}

	inline VectorFloatColor VectorFloatColor::operator * (float right) const
	{
		return operator*(VectorFloat(right));
	}

	inline VectorByte VectorFloatColor::ToByte() const
	{
#ifdef LEGACY
		static const __m128 ZERO = _mm_setzero_ps();
		static const __m128 M255 = _mm_set_ps1(255.0f);

		static const __m128 YRU = _mm_set_ps1(1.402f);
		static const __m128 YBV = _mm_set_ps1(1.772f);
		static const __m128 YGU = _mm_set_ps1(-0.714f);
		static const __m128 YGV = _mm_set_ps1(-0.334f);

		float4 b = y + float4(YBV) * v;
		float4 g = y + float4(YGU) * u + float4(YGV) * v;
		float4 r = y + float4(YRU) * u;

		__m128 vb = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(b.get_value(), M255), ZERO)));
		__m128 vg = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(g.get_value(), M255), ZERO)));
		__m128 vr = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(r.get_value(), M255), ZERO)));

		__m128 bg01 = _mm_unpacklo_ps(vb, vg);
		__m128 ra01 = _mm_unpacklo_ps(vr, ZERO);
		__m128 bg23 = _mm_unpackhi_ps(vb, vg);
		__m128 ra23 = _mm_unpackhi_ps(vr, ZERO);

		__m128i bgra0 = _mm_castps_si128(_mm_shuffle_ps(bg01, ra01, 0x44));
		__m128i bgra1 = _mm_castps_si128(_mm_shuffle_ps(bg01, ra01, 0xEE));
		__m128i bgra2 = _mm_castps_si128(_mm_shuffle_ps(bg23, ra23, 0x44));
		__m128i bgra3 = _mm_castps_si128(_mm_shuffle_ps(bg23, ra23, 0xEE));

		__m128i bgra01 = _mm_packus_epi32(bgra0, bgra1);
		__m128i bgra23 = _mm_packus_epi32(bgra2, bgra3);

		__m128i bgra0123 = _mm_packus_epi16(bgra01, bgra23);

		return byte16(bgra0123);
#else
		static const __m256 ZERO = _mm256_setzero_ps();
		static const __m256 M255 = _mm256_set1_ps(255.0f);

		static const __m256 YRU = _mm256_set1_ps(1.402f);
		static const __m256 YBV = _mm256_set1_ps(1.772f);
		static const __m256 YGU = _mm256_set1_ps(-0.714f);
		static const __m256 YGV = _mm256_set1_ps(-0.334f);

		float8 b = y + float8(YBV) * v;
		float8 g = y + float8(YGU) * u + float8(YGV) * v;
		float8 r = y + float8(YRU) * u;

		__m256 vb = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_max_ps(_mm256_min_ps(b.get_value(), M255), ZERO)));
		__m256 vg = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_max_ps(_mm256_min_ps(g.get_value(), M255), ZERO)));
		__m256 vr = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_max_ps(_mm256_min_ps(r.get_value(), M255), ZERO)));

		__m256 bg01_45 = _mm256_unpacklo_ps(vb, vg);
		__m256 ra01_45 = _mm256_unpacklo_ps(vr, ZERO);
		__m256 bg23_67 = _mm256_unpackhi_ps(vb, vg);
		__m256 ra23_67 = _mm256_unpackhi_ps(vr, ZERO);

		__m256i bgra0_4d = _mm256_castps_si256(_mm256_shuffle_ps(bg01_45, ra01_45, 0x44));
		__m256i bgra1_5d = _mm256_castps_si256(_mm256_shuffle_ps(bg01_45, ra01_45, 0xEE));
		__m256i bgra2_6d = _mm256_castps_si256(_mm256_shuffle_ps(bg23_67, ra23_67, 0x44));
		__m256i bgra3_7d = _mm256_castps_si256(_mm256_shuffle_ps(bg23_67, ra23_67, 0xEE));

		__m128i bgra01 = _mm_packus_epi32(_mm256_extractf128_si256(bgra0_4d, 0), _mm256_extractf128_si256(bgra1_5d, 0));
		__m128i bgra23 = _mm_packus_epi32(_mm256_extractf128_si256(bgra2_6d, 0), _mm256_extractf128_si256(bgra3_7d, 0));
		__m128i bgra45 = _mm_packus_epi32(_mm256_extractf128_si256(bgra0_4d, 1), _mm256_extractf128_si256(bgra1_5d, 1));
		__m128i bgra67 = _mm_packus_epi32(_mm256_extractf128_si256(bgra2_6d, 1), _mm256_extractf128_si256(bgra3_7d, 1));

		__m128i bgra0123 = _mm_packus_epi16(bgra01, bgra23);
		__m128i bgra4567 = _mm_packus_epi16(bgra45, bgra67);

		return byte32(_mm256_set_m128i(bgra4567, bgra0123));
#endif
	}

	// ==================================================================================================
}
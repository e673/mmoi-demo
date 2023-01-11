#pragma once

#ifdef LEGACY
#include <xmmintrin.h>
#else
#include <immintrin.h>
#endif

// #include <zmmintrin.h>

namespace ip
{	
	class float4;	// __m128 type equivalent
	class byte16;	// __m128i type equivalent
	class int4;		// __m128i type equivalent

#ifndef LEGACY

	class float8;	// __m256 type equivalent
	class byte32;	// __m256i type equivalent
	class int8;		// __m256i type equivalent

#endif

#ifdef LEGACY
	typedef float4 VectorFloat;
	typedef byte16 VectorByte;
	typedef int4 VectorInt;
#else
	typedef float8 VectorFloat;
	typedef byte32 VectorByte;
	typedef int8 VectorInt;
#endif

	// ==================================================================================================

	class float4
	{
		__m128 value;

	public:
		float4() = default;

		inline float4(__m128 value)
			: value(value) {}

		inline float4(float value)
			: value(_mm_set_ps1(value)) {}


		inline float4(float v0, float v1, float v2, float v3)
			: value(_mm_set_ps(v3, v2, v1, v0)) {}

		static constexpr int size = 4;

		// ------------------------------------------------------------------------------------------

		inline float4 operator + (float4 right) const
		{
			return float4(_mm_add_ps(value, right.value));
		}

		inline float4 operator - (float4 right) const
		{
			return float4(_mm_sub_ps(value, right.value));
		}

		inline float4 operator * (float4 right) const
		{
			return float4(_mm_mul_ps(value, right.value));
		}

		inline float4 operator / (float4 right) const
		{
			return float4(_mm_div_ps(value, right.value));
		}

		inline float4 operator & (float4 right) const
		{
			return float4(_mm_and_ps(value, right.value));
		}

		inline __m128 get_value() const
		{
			return value;
		}

		// -------------------------------------------------------------------------

		inline float4 abs() const
		{
			static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
			return float4(_mm_and_ps(value, SIGNMASK));
		}

		inline float4 op_max(float4 other) const
		{
			return float4(_mm_max_ps(value, other.value));
		}

		inline float4 op_min(float4 other) const
		{
			return float4(_mm_min_ps(value, other.value));
		}

		inline float get(const int index) const
		{
			return value.m128_f32[index];
		}

		inline void set(const int index, float v)
		{
			value.m128_f32[index] = v;
		}

		// -------------------------------------------------------------------------

		inline static float4 zero()
		{
			return float4(_mm_setzero_ps());
		}

		inline static float4 load(const void *mem)
		{
			return float4(_mm_load_ps(static_cast<const float*>(mem)));
		}

		inline static float4 loadu(const void *mem)
		{
			return float4(_mm_loadu_ps(static_cast<const float*>(mem)));
		}

		inline static float4 load_single(const void *mem)
		{
			return float4(_mm_load_ss(static_cast<const float*>(mem)));
		}

		inline static float4 broadcast(const void *mem)
		{
			return float4(_mm_broadcast_ss(static_cast<const float*>(mem)));
		}

		inline void store(void *mem)
		{
			_mm_store_ps(static_cast<float*>(mem), value);
		}

		inline void stream(void *mem)
		{
			_mm_stream_ps(static_cast<float*>(mem), value);
		}


		// ------------------------------------------------------------------------------------------

		inline float4 unpack_lo(float4 other) const
		{
			return float4(_mm_unpacklo_ps(value, other.value));
		}

		inline float4 unpack_hi(float4 other) const
		{
			return float4(_mm_unpackhi_ps(value, other.value));
		}

		template <int rule>
		inline float4 shuffle(float4 other) const
		{
			return float4(_mm_shuffle_ps(value, other.value, rule));
		}

		inline float4 blend(float4 other, float4 mask) const
		{
			return _mm_blendv_ps(value, other.value, mask.value);
		}

		// ------------------------------------------------------------------------------------------

		inline float4 compare_gt_ord_ns(float4 other) const
		{
			return float4(_mm_cmp_ps(value, other.value, _CMP_GE_OQ));
		}

		inline float4 compare_lt_ord_ns(float4 other) const
		{
			return float4(_mm_cmp_ps(value, other.value, _CMP_LT_OQ));
		}

		// -------------------------------------------------------------------------

#ifndef LEGACY
		inline float8 expand_float8() const;
#endif

		inline byte16 cast_byte16() const;

		// ------------------------------------------------------------------------------------------

		inline int4 conv_i32() const;
		inline int4 cast_int8() const;

	};


	// ==================================================================================================
	
#ifndef LEGACY

	class float8
	{		
		__m256 value;

	public:

		// ------------------------------------------------------------------------------------------

		inline float8() = default;

		inline float8(float value)
			: value(_mm256_set1_ps(value)) {}

		inline float8(__m256 value)
			: value(value) {}

		inline float8(float4 lower, float4 upper)
			: value(lower.expand_float8().insert128_hi(upper).get_value()) {}

		inline float8(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
			: value(_mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0)) {}

		static constexpr int size = 8;

		// ------------------------------------------------------------------------------------------

		inline static float8 zero()
		{
			return float8(_mm256_setzero_ps());
		}

		inline static float8 load(const void *mem)
		{
			return float8(_mm256_load_ps(static_cast<const float*>(mem)));
		}

		inline static float8 loadu(const void *mem)
		{
			return float8(_mm256_load_ps(static_cast<const float*>(mem)));
		}

		inline static float8 broadcast(const void *mem)
		{
			return float8(_mm256_broadcast_ss(static_cast<const float*>(mem)));
		}

		inline void store(void *mem)
		{
			_mm256_store_ps(static_cast<float*>(mem), value);
		}

		inline void stream(void *mem)
		{
			_mm256_stream_ps(static_cast<float*>(mem), value);
		}

		// ------------------------------------------------------------------------------------------

		inline float4 lower() const
		{
			return float4(_mm256_castps256_ps128(value));
		}

		inline float4 upper() const
		{
			return float4(_mm256_extractf128_ps(value, 1));
		}


		inline float8 operator + (float8 right) const
		{
			return float8(_mm256_add_ps(value, right.value));
		}

		inline float8 operator - (float8 right) const
		{
			return float8(_mm256_sub_ps(value, right.value));
		}

		inline float8 operator * (float8 right) const
		{
			return float8(_mm256_mul_ps(value, right.value));
		}

		inline float8 operator / (float8 right) const
		{
			return float8(_mm256_div_ps(value, right.value));
		}

		inline float8 operator & (float8 right) const
		{
			return float8(_mm256_and_ps(value, right.value));
		}

		inline __m256 get_value() const
		{
			return value;
		}

		// ------------------------------------------------------------------------------------------

		inline float8 abs() const
		{
			static const __m256 SIGNMASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
			return float8(_mm256_and_ps(value, SIGNMASK));
		}

		inline float8 op_max(float8 other) const
		{
			return float8(_mm256_max_ps(value, other.value));
		}

		inline float8 op_min(float8 other) const
		{
			return float8(_mm256_min_ps(value, other.value));
		}

		// ------------------------------------------------------------------------------------------

		inline float get(const int index) const
		{
			return value.m256_f32[index];
		}

		inline void set(const int index, float v)
		{
			value.m256_f32[index] = v;
		}

		inline float8 unpack_lo(float8 other) const
		{
			return float8(_mm256_unpacklo_ps(value, other.value));
		}

		inline float8 unpack_hi(float8 other) const
		{
			return float8(_mm256_unpackhi_ps(value, other.value));
		}

		template <int rule>
		inline float8 shuffle(float8 other) const
		{
			return float8(_mm256_shuffle_ps(value, other.value, rule));
		}

		template <int rule>
		inline float8 permute128(float8 other) const
		{
			return float8(_mm256_permute2f128_ps(value, other.value, rule));
		}

		inline float8 insert128_hi(float4 other) const
		{
			return float8(_mm256_insertf128_ps(value, other.get_value(), 1));
		}

		inline float8 blend(float8 other, float8 mask) const
		{
			return _mm256_blendv_ps(value, other.value, mask.value);
		}

		// ------------------------------------------------------------------------------------------

		inline float8 compare_gt_ord_ns(float8 other) const
		{
			return float8(_mm256_cmp_ps(value, other.value, _CMP_GE_OQ));
		}

		inline float8 compare_lt_ord_ns(float8 other) const
		{
			return float8(_mm256_cmp_ps(value, other.value, _CMP_LT_OQ));
		}

		// ------------------------------------------------------------------------------------------

		inline int8 conv_i32() const;
		inline int8 cast_int8() const;
	};

#endif

	// ==================================================================================================

	class byte16
	{
		__m128i value;

	public:
		inline byte16() = default;

		inline byte16(__m128i value)
			: value(value) {}

		inline __m128i get_value() const
		{
			return value;
		}

		inline int4 convert_u32() const;
	};

	// ==================================================================================================

#ifndef LEGACY

	class byte32
	{
		__m256i value;

	public:
		inline byte32() = default;

		inline byte32(__m256i value)
			: value(value) {}

		inline __m256i get_value() const
		{
			return value;
		}
	};

#endif

	// ==================================================================================================

	class int4
	{
		__m128i value;

	public:
		inline int4() = default;

		inline int4(__m128i value)
			: value(value) {}

		inline __m128i get_value() const
		{
			return value;
		}

		// ------------------------------------------------------------------------------------------

		inline int get(const int index) const
		{
			return value.m128i_i32[index];
		}

		// ------------------------------------------------------------------------------------------

		inline float4 convert_f32() const
		{
			return float4(_mm_cvtepi32_ps(value));
		}
	};

	// ==================================================================================================

#ifndef LEGACY

	class int8
	{
		__m256i value;

	public:
		inline int8() = default;

		inline int8(__m256i value)
			: value(value) {}

		inline int8(int value)
			: value(_mm256_set1_epi32(value)) {}

		inline __m256i get_value() const
		{
			return value;
		}

		// ------------------------------------------------------------------------------------------

		inline int get(const int index) const
		{
			return value.m256i_i32[index];
		}

		// ------------------------------------------------------------------------------------------

		inline float8 convert_f32() const
		{
			return float8(_mm256_cvtepi32_ps(value));
		}

		inline float8 cast_float8() const
		{
			return float8(_mm256_castsi256_ps(value));
		}

		// ------------------------------------------------------------------------------------------

		inline int8 blend(int8 other, float8 mask) const
		{
			return cast_float8().blend(other.cast_float8(), mask).cast_int8();
		}

		// ------------------------------------------------------------------------------------------

		inline int8 operator & (int8 other) const
		{
			return int8(_mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(other.value))));
		}
	};

#endif
	// ==================================================================================================

#ifndef LEGACY

	float8 float4::expand_float8() const
	{
		return float8(_mm256_castps128_ps256(value));
	}

#endif

	byte16 float4::cast_byte16() const
	{
		return byte16(_mm_castps_si128(value));
	}

	inline int4 byte16::convert_u32() const
	{
		return int4(_mm_cvtepu8_epi32(value));
	}

	inline int4 float4::conv_i32() const
	{
		return int4(_mm_cvtps_epi32(value));
	}

	inline int4 float4::cast_int8() const
	{
		return int4(_mm_castps_si128(value));
	}


#ifndef LEGACY

	inline int8 float8::conv_i32() const
	{
		return int8(_mm256_cvtps_epi32(value));
	}

	inline int8 float8::cast_int8() const
	{
		return int8(_mm256_castps_si256(value));
	}

#endif

}

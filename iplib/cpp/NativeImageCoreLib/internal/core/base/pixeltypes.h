#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

namespace ip
{
	byte i2b(int x);
	byte f2b(float x);

	constexpr float qRed = 0.299f;
	constexpr float qGreen = 0.587f;
	constexpr float qBlue = 0.114f;

#pragma pack(push, 1)

	template <int N>
	class PixelUntyped;

	template <class T, int N>
	class PixelTyped;

	// inner type: unsigned char, 3 bpp
	class PixelByteRGB;

	// inner type: unsigned char, 4 bpp
	class PixelByteRGB4;

	// inner type: unsigned char, 4 bpp
	class PixelByteRGBA;

	// inner type: float, 12 bpp
	class PixelFloatRGB;

	// inner type: float, 16 bpp
	class PixelFloatRGB4;

	// inner type: float, 16 bpp
	class PixelFloatRGBA;

	// inner type: float, 8 bpp
	class PixelFloatComplex;

	// inner type: float, 32 bpp
	class PixelFloat256;

	// inner type: float, 64 bpp
	class PixelFloat256x2;

	template <class T>
	class PixelVector;

	// ==================================================================================================
	//                                       Type implicit conversion rules               
	// ==================================================================================================
	/*
		
		PixelByteRGBA    PixelFloatRGBA
		     |                |
		PixelByteRGB4    PixelFloatRGB4
		     |                |
		PixelByteRGB     PixelFloatRGB
		     ^                ^
		PixelByte     -> PixelFloat
	*/

	// ==================================================================================================
	//                                        PixelUntyped               
	// ==================================================================================================

	template <int N>
	class PixelUntyped
	{
	public:
		char data[N];
	};

	// ==================================================================================================
	//                                          PixelTyped
	// ==================================================================================================

	template <class T, int N>
	class PixelTyped
	{
	public:
		T data[N];
	};

	// ==================================================================================================
	//                                          PixelVector
	// ==================================================================================================

	template <class T>
	class PixelVector
	{
	public:
		T x, y;

	public:
		PixelVector(T x, T y)
			: x(x), y(y) {}
	};


	// ==================================================================================================
	//                                            PixelByteRGB             
	// ==================================================================================================

	class PixelByteRGB
	{
	public:
		byte b;
		byte g;
		byte r;

		PixelByteRGB() = default;
		PixelByteRGB(const PixelByteRGB&) = default;

		PixelByteRGB(byte c);
		PixelByteRGB(byte r, byte g, byte b);
		explicit PixelByteRGB(const PixelFloatRGB &other);

		PixelByteRGB& operator += (PixelByteRGB other);
		PixelByteRGB& operator -= (PixelByteRGB other);

		float ToGray() const;

		explicit operator byte() const;
		explicit operator float() const;
	};

	typedef PixelByteRGB PixelB3;

	// ==================================================================================================
	//                                         PixelByteRGB4               
	// ==================================================================================================


	class PixelByteRGB4 :
		public PixelByteRGB
	{
		byte unused;

	public:
		PixelByteRGB4() = default;
		PixelByteRGB4(const PixelByteRGB4&) = default;

		PixelByteRGB4(byte c);
		PixelByteRGB4(byte r, byte g, byte b);
		PixelByteRGB4(const PixelByteRGB& other);
	};

	typedef PixelByteRGB4 PixelB4a;

	// ==================================================================================================
	//                                         PixelByteRGBA
	// ==================================================================================================

	class PixelByteRGBA :
		public PixelByteRGB
	{
	public:
		byte a;

		PixelByteRGBA() = default;
		PixelByteRGBA(const PixelByteRGBA&) = default;

		PixelByteRGBA(byte c);
		PixelByteRGBA(byte r, byte g, byte b, byte a = 0);
		PixelByteRGBA(const PixelByteRGB& other);
	};

	typedef PixelByteRGBA PixelB4;

	PixelFloatRGBA operator * (PixelByteRGBA a, float q);
	PixelFloatRGBA operator * (float q, PixelByteRGBA a);

	// ==================================================================================================
	//                                            PixelFloatRGB               
	// ==================================================================================================

	class PixelFloatRGB
	{
	public:
		float b;
		float g;
		float r;

		PixelFloatRGB() = default;
		PixelFloatRGB(const PixelFloatRGB&) = default;

		PixelFloatRGB(float c);
		PixelFloatRGB(float r, float g, float b);
		PixelFloatRGB(const PixelByteRGB& other);

		float ToGray() const;

		PixelFloatRGB& operator += (PixelFloatRGB other);
		PixelFloatRGB& operator -= (PixelFloatRGB other);
	};

	PixelFloatRGB operator + (PixelFloatRGB x, PixelFloatRGB y);
	PixelFloatRGB operator - (PixelFloatRGB x, PixelFloatRGB y);
	PixelFloatRGB operator * (PixelFloatRGB x, float y);
	PixelFloatRGB operator * (float x, PixelFloatRGB y);
	PixelFloatRGB operator / (PixelFloatRGB x, float y);
	bool operator == (PixelFloatRGB x, PixelFloatRGB y);
	bool operator != (PixelFloatRGB x, PixelFloatRGB y);

	typedef PixelFloatRGB ColorF3;

	//------------------------------------------------------------------------

	class PixelFloatRGB4 :
		public PixelFloatRGB
	{
		float unused;

	public:
		PixelFloatRGB4() = default;
		PixelFloatRGB4(const PixelFloatRGB4&) = default;

		PixelFloatRGB4(float c);
		PixelFloatRGB4(float r, float g, float b);
		PixelFloatRGB4(const PixelFloatRGB& other);

		PixelFloatRGB4& operator += (PixelFloatRGB4 other);
		PixelFloatRGB4& operator -= (PixelFloatRGB4 other);

		PixelFloatRGB4& operator *= (float other);

		explicit operator float() const;

	};

	typedef PixelFloatRGB4 ColorF4a;

	//------------------------------------------------------------------------

	class PixelFloatRGBA :
		public PixelFloatRGB
	{
	public:
		float a;

		PixelFloatRGBA() = default;
		PixelFloatRGBA(const PixelFloatRGBA&) = default;

		inline PixelFloatRGBA(float c);
		inline PixelFloatRGBA(float r, float g, float b, float a = 0.0f);
		inline PixelFloatRGBA(const PixelFloatRGB& other);
		inline PixelFloatRGBA(const PixelByteRGB& other);

		PixelFloatRGBA& operator += (const PixelFloatRGBA &other);
		PixelFloatRGBA& operator /= (float v);

		explicit operator float() const;
		explicit operator PixelByteRGBA() const;
	};

	PixelFloatRGBA operator + (PixelFloatRGBA x, PixelFloatRGBA y);
	PixelFloatRGBA operator - (PixelFloatRGBA x, PixelFloatRGBA y);
	PixelFloatRGBA operator * (PixelFloatRGBA x, float y);
	PixelFloatRGBA operator * (float x, PixelFloatRGBA y);
	PixelFloatRGBA operator / (PixelFloatRGBA x, float y);
	bool operator == (PixelFloatRGBA x, PixelFloatRGBA y);
	bool operator != (PixelFloatRGBA x, PixelFloatRGBA y);

	typedef PixelFloatRGBA ColorF4;

	//------------------------------------------------------------------------

	class PixelFloatComplex
	{
	public:
		float Re, Im;

		PixelFloatComplex() = default;
		PixelFloatComplex(const PixelFloatComplex&) = default;

		PixelFloatComplex(float Re, float Im);
	};

	PixelFloatComplex operator + (PixelFloatComplex a, PixelFloatComplex b);
	PixelFloatComplex operator - (PixelFloatComplex a, PixelFloatComplex b);
	PixelFloatComplex operator * (PixelFloatComplex a, PixelFloatComplex b);

	//------------------------------------------------------------------------

	class PixelFloatVector
	{
	public:
		float x, y;

		PixelFloatVector() = default;
		PixelFloatVector(const PixelFloatVector&) = default;

		PixelFloatVector(float x, float y);

		float Norm() const;

		PixelFloatVector& operator += (PixelFloatVector other);
	};

	PixelFloatVector operator + (PixelFloatVector a, PixelFloatVector b);
	PixelFloatVector operator - (PixelFloatVector a, PixelFloatVector b);
	PixelFloatVector operator * (PixelFloatVector a, float q);
	PixelFloatVector operator * (float q, PixelFloatVector a);

	//------------------------------------------------------------------------

	class PixelFloat256
	{
	public:
		float v[8];
	};

	//------------------------------------------------------------------------

	class PixelFloat256x2
	{
	public:
		PixelFloat256 x, y;
	};

	//////////////////////////////////////////////////////////////////////////

	inline PixelFloatRGBA::PixelFloatRGBA(float c)
		: PixelFloatRGB(c), a(0.0f) {}

	inline PixelFloatRGBA::PixelFloatRGBA(float r, float g, float b, float a)
		: PixelFloatRGB(r, g, b), a(a) {}

	inline PixelFloatRGBA::PixelFloatRGBA(const PixelFloatRGB& other)
		: PixelFloatRGB(other), a(0.0f) {}

	inline PixelFloatRGBA::PixelFloatRGBA(const PixelByteRGB& other)
		: PixelFloatRGB(other), a(0.0f) {}

#pragma pack(pop)

}
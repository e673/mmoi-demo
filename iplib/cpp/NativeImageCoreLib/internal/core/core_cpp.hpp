#include <malloc.h>

#include "../../iplib/common.h"
#include "../../iplib/image/core.h"
#include "../../iplib/image/core3d.h"

// ==================================================================================================
//                                    BitmapDataStructure               
// ==================================================================================================

// #include "bitmap/bitmapdatastructure.h"

namespace ip
{
	BitmapDataStructure BitmapDataStructure::Fragment(const BitmapDataStructure &img, int OffsetX, int OffsetY, int Width, int Height, int PixelSize)
	{
		check(Width >= 0 && Height >= 0);

		ptrdiff_t diff = (size_t)data - (size_t)img.data;
		ptrdiff_t ofsy = diff / stride;
		ptrdiff_t ofsx = diff - ofsy * stride;

		ofsx += OffsetX * PixelSize;
		ofsy += OffsetY;

		check(ofsx >= 0 && ofsx + Width * PixelSize <= stride);
		check(ofsy >= 0 && ofsy + Height <= height);

		BitmapDataStructure res;
		res.data = (byte*)data + stride * ofsy + ofsx;
		res.width = Width;
		res.height = Height;
		res.stride = stride;

		return res;
	}

	BitmapDataStructure3D BitmapDataStructure3D::Fragment(const BitmapDataStructure3D &img, int OffsetX, int OffsetY, int OffsetZ, int SizeX, int SizeY, int SizeZ, int PixelSize)
	{
		check(SizeX >= 0 && SizeY >= 0 && SizeZ >= 0);

		ptrdiff_t diff = (size_t)data - (size_t)img.data;
		ptrdiff_t ofsz = diff / stride_z;
		ptrdiff_t ofsy = (diff - ofsz * stride_z) / stride_y;
		ptrdiff_t ofsx = diff - ofsz * stride_z - ofsy * stride_y;

		ofsx += OffsetX * PixelSize;
		ofsy += OffsetY;
		ofsz += OffsetZ;

		check(ofsx >= 0 && ofsx + SizeX * PixelSize <= stride_y);
		check(ofsy >= 0 && ofsy + SizeY <= size_y);
		check(ofsz >= 0 && ofsz + SizeZ <= size_z);

		BitmapDataStructure3D res;
		res.data = (byte*)data + ofsz * stride_z + ofsy * stride_y + ofsx;
		res.size_x = SizeX;
		res.size_y = SizeY;
		res.size_z = SizeZ;
		res.stride_y = stride_y;
		res.stride_z = stride_z;

		return res;
	}
}

// ==================================================================================================
//                                           Image               
// ==================================================================================================

// #include "bitmap/image.h"
// #include "bitmap/image3d.h"

namespace ip
{
	namespace internal
	{
		BitmapData::BitmapData(int Width, int Height, int PixelSize)
		{
 			check(PixelSize > 0 && Width > 0 && Height > 0);

			width = Width;
			height = Height;
			stride = (Width * PixelSize + Alignment - 1) / Alignment * Alignment;

			size_t AllocSize = stride * Height;

			data = _aligned_malloc(AllocSize, Alignment);

			check(data != nullptr);
		}

		BitmapData::~BitmapData()
		{
			check(data != nullptr);
			_aligned_free(data);
		}

		// ---------------------------------------------------------------------

		BitmapData3D::BitmapData3D(int SizeX, int SizeY, int SizeZ, int PixelSize)
		{
			check(PixelSize > 0 && SizeX > 0 && SizeY > 0 && SizeZ > 0);

			size_x = SizeX;
			size_y = SizeY;
			size_z = SizeZ;

			stride_y = (SizeX * PixelSize + Alignment - 1) / Alignment * Alignment;
			stride_z = stride_y * SizeY;

			size_t AllocSize = stride_z * SizeZ;

			data = _aligned_malloc(AllocSize, Alignment);

			check(data != nullptr);
		}

		BitmapData3D::~BitmapData3D()
		{
			check(data != nullptr);
			_aligned_free(data);
		}
	}
}

// ==================================================================================================
//                                          PixelTypes               
// ==================================================================================================

// #include "base/pixeltypes.h"

namespace ip
{
	byte i2b(int x)
	{
		return (x > 255) ? 255 : (x < 0 ? 0 : (byte)x);
	}

	byte f2b(float x)
	{
		return (x >= 255.0f) ? 255 : (x <= 0.0f ? 0 : (byte)(x + 0.5f));
	}

	PixelByteRGB::PixelByteRGB(byte c)
		: r(c), g(c), b(c) {}

	PixelByteRGB::PixelByteRGB(byte r, byte g, byte b)
		: r(r), g(g), b(b) {}

	PixelByteRGB::PixelByteRGB(const PixelFloatRGB &other)
		: r(f2b(other.r)), g(f2b(other.g)), b(f2b(other.b)) {}

	PixelByteRGB& PixelByteRGB::operator += (PixelByteRGB other)
	{
		r += other.r;
		g += other.g;
		b += other.b;
		return *this;
	}

	PixelByteRGB& PixelByteRGB::operator -= (PixelByteRGB other)
	{
		r -= other.r;
		g -= other.g;
		b -= other.b;
		return *this;
	}

	PixelByteRGB::operator byte() const
	{
		return f2b(r * 0.299f + g * 0.587f + b * 0.114f);
	}

	PixelByteRGB::operator float() const
	{
		return r * 0.299f + g * 0.587f + b * 0.114f;
	}
	// ---------------------------------------------------------------------------------------------

	PixelByteRGB4::PixelByteRGB4(byte c)
		: PixelByteRGB(c) {}

	PixelByteRGB4::PixelByteRGB4(byte r, byte g, byte b)
		: PixelByteRGB(r, g, b) {}

	PixelByteRGB4::PixelByteRGB4(const PixelByteRGB& other)
		: PixelByteRGB(other) {}

	// ---------------------------------------------------------------------------------------------

	PixelByteRGBA::PixelByteRGBA(byte c)
		: PixelByteRGB(c), a(0) {}

	PixelByteRGBA::PixelByteRGBA(byte r, byte g, byte b, byte a)
		: PixelByteRGB(r, g, b), a(a) {}

	PixelByteRGBA::PixelByteRGBA(const PixelByteRGB& other)
		: PixelByteRGB(other), a(0) {}

	PixelFloatRGBA operator * (PixelByteRGBA a, float q)
	{
		return PixelFloatRGBA(a.r * q, a.g * q, a.b * q, a.a * q);
	}

	PixelFloatRGBA operator * (float q, PixelByteRGBA a)
	{
		return PixelFloatRGBA(a.r * q, a.g * q, a.b * q, a.a * q);
	}

	// ---------------------------------------------------------------------------------------------

	PixelFloatRGB::PixelFloatRGB(float c)
		: r(c), g(c), b(c) {}

	PixelFloatRGB::PixelFloatRGB(float r, float g, float b)
		: r(r), g(g), b(b) {}

	PixelFloatRGB::PixelFloatRGB(const PixelByteRGB& other)
		: r(other.r), g(other.g), b(other.b) {}


	float PixelFloatRGB::ToGray() const
	{
		return 0.299f * r + 0.587f * g + 0.114f * b;
	}

	PixelFloatRGB& PixelFloatRGB::operator += (PixelFloatRGB other)
	{
		r += other.r;
		g += other.g;
		b += other.b;
		return *this;
	}

	PixelFloatRGB& PixelFloatRGB::operator -= (PixelFloatRGB other)
	{
		r -= other.r;
		g -= other.g;
		b -= other.b;
		return *this;
	}


	PixelFloatRGB operator + (PixelFloatRGB x, PixelFloatRGB y)
	{
		return PixelFloatRGB(x.r + y.r, x.g + y.g, x.b + y.b);
	}

	PixelFloatRGB operator - (PixelFloatRGB x, PixelFloatRGB y)
	{
		return PixelFloatRGB(x.r - y.r, x.g - y.g, x.b - y.b);
	}

	PixelFloatRGB operator * (PixelFloatRGB x, float y)
	{
		return PixelFloatRGB(x.r * y, x.g * y, x.b * y);
	}

	PixelFloatRGB operator * (float x, PixelFloatRGB y)
	{
		return PixelFloatRGB(y.r * x, y.g * x, y.b * x);
	}

	PixelFloatRGB operator / (PixelFloatRGB x, float y)
	{
		return PixelFloatRGB(x.r / y, x.g / y, x.b / y);
	}

	bool operator == (PixelFloatRGB x, PixelFloatRGB y)
	{
		return x.r == y.r && x.g == y.g && x.b == y.b;
	}

	bool operator != (PixelFloatRGB x, PixelFloatRGB y)
	{
		return x.r != y.r || x.g != y.g || x.b != y.b;
	}

	// ---------------------------------------------------------------------------------------------

	PixelFloatRGB4::PixelFloatRGB4(float c)
		: PixelFloatRGB(c) {}

	PixelFloatRGB4::PixelFloatRGB4(float r, float g, float b)
		: PixelFloatRGB(r, g, b) {}

	PixelFloatRGB4::PixelFloatRGB4(const PixelFloatRGB& other)
		: PixelFloatRGB(other) {}

	PixelFloatRGB4& PixelFloatRGB4::operator += (PixelFloatRGB4 other)
	{
		r += other.r;
		g += other.g;
		b += other.b;
		return *this;
	}

	PixelFloatRGB4& PixelFloatRGB4::operator -= (PixelFloatRGB4 other)
	{
		r -= other.r;
		g -= other.g;
		b -= other.b;
		return *this;
	}

	PixelFloatRGB4& PixelFloatRGB4::operator *= (float other)
	{
		r *= other;
		g *= other;
		b *= other;
		return *this;
	}

	PixelFloatRGB4::operator float() const
	{
		return ToGray();
	}


	// ---------------------------------------------------------------------------------------------

	PixelFloatRGBA& PixelFloatRGBA::operator += (const PixelFloatRGBA &other)
	{
		r += other.r;
		g += other.g;
		b += other.b;
		a += other.a;
		return *this;
	}

	PixelFloatRGBA& PixelFloatRGBA::operator /= (float v)
	{
		float q = 1.0f / v;
		r *= q;
		g *= q;
		b *= q;
		a *= q;
		return *this;
	}

	PixelFloatRGBA::operator float() const
	{
		return qRed * r + qGreen * g + qBlue * b;
	}

	PixelFloatRGBA::operator PixelByteRGBA() const
	{
		return PixelByteRGBA(f2b(r), f2b(g), f2b(b), f2b(a));
	}

	PixelFloatRGBA operator + (PixelFloatRGBA x, PixelFloatRGBA y)
	{
		return PixelFloatRGBA(x.r + y.r, x.g + y.g, x.b + y.b, x.a + y.a);
	}

	PixelFloatRGBA operator - (PixelFloatRGBA x, PixelFloatRGBA y)
	{
		return PixelFloatRGBA(x.r - y.r, x.g - y.g, x.b - y.b, x.a - y.a);
	}

	PixelFloatRGBA operator * (PixelFloatRGBA x, float y)
	{
		return PixelFloatRGBA(x.r * y, x.g * y, x.b * y, x.a * y);
	}

	PixelFloatRGBA operator * (float x, PixelFloatRGBA y)
	{
		return PixelFloatRGBA(y.r * x, y.g * x, y.b * x, y.a * x);
	}

	PixelFloatRGBA operator / (PixelFloatRGBA x, float y)
	{
		return PixelFloatRGBA(x.r / y, x.g / y, x.b / y, x.a / y);
	}

	bool operator == (PixelFloatRGBA x, PixelFloatRGBA y)
	{
		return x.r == y.r && x.g == y.g && x.b == y.b && x.a == y.a;
	}

	bool operator != (PixelFloatRGBA x, PixelFloatRGBA y)
	{
		return x.r != y.r || x.g != y.g || x.b != y.b || x.a != y.a;
	}


	// ---------------------------------------------------------------------------------------------

	PixelFloatComplex::PixelFloatComplex(float Re, float Im)
		: Re(Re), Im(Im) {}

	PixelFloatComplex operator + (PixelFloatComplex a, PixelFloatComplex b)
	{
		return PixelFloatComplex(a.Re + b.Re, a.Im + b.Im);
	}

	PixelFloatComplex operator - (PixelFloatComplex a, PixelFloatComplex b)
	{
		return PixelFloatComplex(a.Re - b.Re, a.Im - b.Im);
	}

	PixelFloatComplex operator * (PixelFloatComplex a, PixelFloatComplex b)
	{
		return PixelFloatComplex(a.Re * b.Re - a.Im * b.Im, a.Re * b.Im + a.Im * b.Re);
	}

	// ---------------------------------------------------------------------------------------------

	PixelFloatVector::PixelFloatVector(float x, float y)
		: x(x), y(y) {}

	float PixelFloatVector::Norm() const
	{
		return sqrtf(x * x + y * y);
	}

	PixelFloatVector operator + (PixelFloatVector a, PixelFloatVector b)
	{
		return PixelFloatVector(a.x + b.x, a.y + b.y);
	}

	PixelFloatVector operator - (PixelFloatVector a, PixelFloatVector b)
	{
		return PixelFloatVector(a.x - b.x, a.y - b.y);
	}

	PixelFloatVector operator * (PixelFloatVector a, float q)
	{
		return PixelFloatVector(a.x * q, a.y * q);
	}

	PixelFloatVector operator * (float q, PixelFloatVector a)
	{
		return PixelFloatVector(a.x * q, a.y * q);
	}

	PixelFloatVector& PixelFloatVector::operator += (PixelFloatVector other)
	{
		x += other.x;
		y += other.y;
		return *this;
	}
}

// ==================================================================================================
//                                         ImageIO               
// ==================================================================================================

#include "../../iplib/image/io/imageio.h"

#pragma comment(lib, "GdiPlus.lib")

namespace ip
{
	namespace internal
	{
		LockedBitmap::LockedBitmap(Gdiplus::Bitmap &B, Gdiplus::PixelFormat pf, int pixelsize)
			: B(B)
		{
			Init(pf, pixelsize);
		}

		LockedBitmap::LockedBitmap(Gdiplus::Bitmap &B)
			: B(B)
		{
			Init(PixelFormat24bppRGB, 4);
		}

		LockedBitmap::~LockedBitmap()
		{
			B.UnlockBits(&bitmapdata);
		}

		void LockedBitmap::Init(Gdiplus::PixelFormat pf, int pixelsize)
		{
			Gdiplus::RectF boundsF;
			Gdiplus::Unit unit = Gdiplus::UnitPixel;
			B.GetBounds(&boundsF, &unit);

			Gdiplus::Rect bounds((int)boundsF.X, (int)boundsF.Y, (int)boundsF.Width, (int)boundsF.Height);
			B.LockBits(&bounds, Gdiplus::ImageLockModeRead | Gdiplus::ImageLockModeWrite, pf, &bitmapdata);
		}

		GdiPlusInitializer GdiPlusInitializer::single_instance;

		GdiPlusInitializer::GdiPlusInitializer()
		{
			Gdiplus::GdiplusStartupInput gdiplusStartupInput;
			Gdiplus::GdiplusStartup(&m_gdiplusToken, &gdiplusStartupInput, NULL);
#ifdef DEBUG
			printf("GDI+ has been initialized\n");
#endif
		}

		GdiPlusInitializer::~GdiPlusInitializer()
		{
			Gdiplus::GdiplusShutdown(m_gdiplusToken);
#ifdef DEBUG
			printf("GDI+ has been released\n");
#endif
		}
	}

	int ImageIO::GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
	{
		UINT  num = 0;          // number of image encoders
		UINT  size = 0;         // size of the image encoder array in bytes

		Gdiplus::ImageCodecInfo* pImageCodecInfo = NULL;

		Gdiplus::GetImageEncodersSize(&num, &size);
		if (size == 0)
			return -1;  // Failure

		pImageCodecInfo = (Gdiplus::ImageCodecInfo*)(malloc(size));
		if (pImageCodecInfo == NULL)
			return -1;  // Failure

		GetImageEncoders(num, size, pImageCodecInfo);

		for (UINT j = 0; j < num; ++j)
		{
			if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
			{
				*pClsid = pImageCodecInfo[j].Clsid;
				free(pImageCodecInfo);
				return j;  // Success
			}
		}

		free(pImageCodecInfo);
		return -1;  // Failure
	}
}

// ==================================================================================================
//                                        DIBitmap               
// ==================================================================================================

#include "../../iplib/image/io/dibitmap.h"

namespace ip
{
	namespace internal
	{
		DIBitmapData::DIBitmapData(int Width, int Height)
		{
			this->width = Width;
			this->height = Height;
			this->stride = Width * 4;

			BITMAPINFO bmi;

			bmi.bmiHeader.biSize = sizeof(BITMAPINFO);
			bmi.bmiHeader.biHeight = -Height;
			bmi.bmiHeader.biWidth = Width;
			bmi.bmiHeader.biPlanes = 1;
			bmi.bmiHeader.biBitCount = 32;
			bmi.bmiHeader.biCompression = 0;  // BI_RGB;
			bmi.bmiHeader.biSizeImage = Height * Width * 4;
			bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 96;
			bmi.bmiHeader.biClrUsed = 0;
			bmi.bmiHeader.biClrImportant = 0;
			bmi.bmiColors[0] = { 0, 0, 0, 0 };

			hdc = CreateCompatibleDC(nullptr);
			hbitmap = CreateDIBSection(nullptr, &bmi, 0, &this->data, nullptr, 0);
			hold = SelectObject(hdc, hbitmap);
		}

		DIBitmapData::~DIBitmapData()
		{
			SelectObject(hdc, hold);
			DeleteObject(hbitmap);
			DeleteDC(hdc);
		}
	}

	DIBitmap::DIBitmap(int Width, int Height)
	{
		base_data.reset(new internal::DIBitmapData(Width, Height));
		data = *base_data;
	}

	/*
	DIBitmap::DIBitmap(DIBitmap &ref, int OffsetX, int OffsetY, int Width, int Height)
	{
		data = ref.base_data->Fragment(ref.data, OffsetX, OffsetY, Width, Height, 4);
		base_data = ref.base_data;
	}
	*/
}

// ==================================================================================================
//                                         VideoIO               
// ==================================================================================================

#include "../../iplib/image/io/videoio.h"

#pragma comment(lib, "strmiids.lib")

namespace ip
{
	namespace internal
	{
		SVideoProcessAdapter::SVideoProcessAdapter(AviReader *pAviReader)
		{
			this->pAviReader = pAviReader;
		}

		SVideoProcessAdapter::~SVideoProcessAdapter(void)
		{

		}

		STDMETHODIMP SVideoProcessAdapter::SampleCB(double, IMediaSample *)
		{
			return S_OK;
		}

		STDMETHODIMP SVideoProcessAdapter::QueryInterface(REFIID iid, LPVOID *ppv)
		{
			if (iid == IID_ISampleGrabberCB || iid == IID_IUnknown)
			{
				*ppv = (void *) static_cast<ISampleGrabberCB*>(this);
				return NOERROR;
			}
			return E_NOINTERFACE;
		}

		STDMETHODIMP_(ULONG) SVideoProcessAdapter::AddRef()
		{
			return 2;
		}

		STDMETHODIMP_(ULONG) SVideoProcessAdapter::Release()
		{
			return 1;
		}

		STDMETHODIMP SVideoProcessAdapter::BufferCB(double SampleTime, BYTE *pBuffer, long nBufferLen)
		{
			pAviReader->OnBufferCB(SampleTime, pBuffer, nBufferLen);
			return S_OK;
		}
	}

	//////////////////////////////////////////////////////////////////////////

	HRESULT	AviReader::GetPin(IBaseFilter *pFilter, PIN_DIRECTION PinDir, IPin **ppPin)
	{
		IEnumPins  *pEnum;
		IPin       *pPin;
		pFilter->EnumPins(&pEnum);
		while (pEnum->Next(1, &pPin, 0) == S_OK)
		{
			PIN_DIRECTION PinDirThis;
			pPin->QueryDirection(&PinDirThis);
			if (PinDir == PinDirThis)
			{
				pEnum->Release();
				*ppPin = pPin;
				return S_OK;
			}
			pPin->Release();
		}
		pEnum->Release();
		return E_FAIL;
	}

	HRESULT AviReader::ConnectFilters(IGraphBuilder *pGraph, IBaseFilter *pFirst, IBaseFilter *pSecond)
	{
		IPin *pOut = NULL, *pIn = NULL;
		HRESULT hr = GetPin(pSecond, PINDIR_INPUT, &pIn);
		if (FAILED(hr)) return hr;
		// The previous filter may have multiple outputs, so try each one!
		IEnumPins  *pEnum;
		pFirst->EnumPins(&pEnum);
		while (pEnum->Next(1, &pOut, nullptr) == S_OK)
		{
			PIN_DIRECTION PinDirThis;
			pOut->QueryDirection(&PinDirThis);
			if (PINDIR_OUTPUT == PinDirThis)
			{
				hr = pGraph->Connect(pOut, pIn);
				if (!FAILED(hr))
					break;
			}
			SafeRelease(&pOut);
		}
		SafeRelease(&pOut);
		SafeRelease(&pEnum);
		SafeRelease(&pIn);
		return hr;
	}

	void AviReader::FreeMediaType(AM_MEDIA_TYPE& mt)
	{
		if (mt.cbFormat != 0)
		{
			CoTaskMemFree((PVOID)mt.pbFormat);
			mt.cbFormat = 0;
			mt.pbFormat = NULL;
		}
		if (mt.pUnk != NULL)
		{
			// pUnk should not be used.
			mt.pUnk->Release();
			mt.pUnk = NULL;
		}
	}

	HRESULT AviReader::Init()
	{
		pGraph = NULL;
		pControl = NULL;
		pEvent = NULL;
		pEventSink = NULL;
		pGrabberF = NULL;
		pGrabber = NULL;
		pSourceF = NULL;
		pEnum = NULL;
		pPin = NULL;
		pNullF = NULL;
		pBuffer = NULL;
		pSeeking = NULL;
		Finished = false;

		HRESULT hr = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pGraph));
		if (FAILED(hr))
			return hr;

		hr = pGraph->QueryInterface(IID_PPV_ARGS(&pControl));
		if (FAILED(hr))
			return hr;

		hr = pGraph->QueryInterface(IID_PPV_ARGS(&pEvent));
		if (FAILED(hr))
			return hr;

		hr = pGraph->QueryInterface(IID_PPV_ARGS(&pEventSink));
		if (FAILED(hr))
			return hr;

		// Create the Sample Grabber filter.
		hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pGrabberF));
		if (FAILED(hr))
			return hr;

		hr = pGraph->AddFilter(pGrabberF, L"Sample Grabber");
		if (FAILED(hr))
			return hr;

		hr = pGrabberF->QueryInterface(IID_ISampleGrabber, (void**)&pGrabber);
		if (FAILED(hr))
			return hr;

		// Configure the media type output
		AM_MEDIA_TYPE mt;
		ZeroMemory(&mt, sizeof(mt));
		mt.majortype = MEDIATYPE_Video;
		mt.subtype = MEDIASUBTYPE_RGB32;

		hr = pGrabber->SetMediaType(&mt);
		if (FAILED(hr))
			return hr;

		hr = pGraph->AddSourceFilter(pszVideoFile, L"Source", &pSourceF);
		if (FAILED(hr))
			return hr;

		hr = ConnectFilters(pGraph, pSourceF, pGrabberF);
		if (FAILED(hr))
			return hr;

		hr = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pNullF));
		if (FAILED(hr))
			return hr;

		hr = pGraph->AddFilter(pNullF, L"Null Filter");
		if (FAILED(hr))
			return hr;

		hr = ConnectFilters(pGraph, pGrabberF, pNullF);
		if (FAILED(hr))
			return hr;

		// After the first sample, the Sample Grabber continues to process samples. This is the default behavior.
		// hr = pGrabber->SetOneShot(TRUE);
		hr = pGrabber->SetOneShot(FALSE);
		if (FAILED(hr))
			return hr;

		// Set modes
		hr = pGrabber->SetBufferSamples(FALSE); // Buffer seems to be no use in callback mode
		if (FAILED(hr))
			return hr;

		// Set callback, 1 = BufferCB
		hr = pGrabber->SetCallback(&adapter, 1);
		if (FAILED(hr))
			return hr;

		// Necessary interfaces for controlling
		hr = pGraph->QueryInterface(IID_IMediaControl, (void**)(&pControl));
		if (FAILED(hr)) return hr;
		hr = pGraph->QueryInterface(IID_IMediaEventEx, (void**)(&pEvent));
		if (FAILED(hr)) return hr;
		hr = pGraph->QueryInterface(IID_IMediaSeeking, (void**)(&pSeeking));
		if (FAILED(hr)) return hr;

#ifdef DEBUG
		LONGLONG duration;
		double rate;
		pSeeking->GetRate(&rate);
		pSeeking->GetDuration(&duration);
		printf("Rate is %lf, duration is %lld", rate, duration);
#endif

		// Turn off the sync clock for max speed
		IMediaFilter *pMediaFilter = NULL;
		hr = pGraph->QueryInterface(IID_IMediaFilter, (void**)(&pMediaFilter));
		if (FAILED(hr)) return hr;
		hr = pMediaFilter->SetSyncSource(NULL);
		if (FAILED(hr)) return hr;
		SafeRelease(&pMediaFilter);

		// Query the media information
		pGrabber->GetConnectedMediaType(&mt);
		if ((mt.formattype == FORMAT_VideoInfo) &&
			(mt.cbFormat >= sizeof(VIDEOINFOHEADER)) &&
			(mt.pbFormat != NULL))
		{
			VIDEOINFOHEADER *pVih = (VIDEOINFOHEADER*)mt.pbFormat;
			this->_Width = pVih->bmiHeader.biWidth;
			this->_Height = pVih->bmiHeader.biHeight;
			this->_BitsPerPixel = pVih->bmiHeader.biBitCount;
		}
		else
			return -1;

		return S_OK;
	}

	HRESULT AviReader::InitWebcam()
	{
		pGraph = NULL;
		pControl = NULL;
		pEvent = NULL;
		pEventSink = NULL;
		pGrabberF = NULL;
		pGrabber = NULL;
		pSourceF = NULL;
		pEnum = NULL;
		pPin = NULL;
		pNullF = NULL;
		pBuffer = NULL;
		pSeeking = NULL;
		pDevEnum = NULL;
		pEnumMoniker = NULL;
		pMoniker = NULL;
		Finished = false;

		HRESULT hr = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pGraph));
		if (FAILED(hr))
			return hr;

		hr = pGraph->QueryInterface(IID_PPV_ARGS(&pControl));
		if (FAILED(hr))
			return hr;

		hr = pGraph->QueryInterface(IID_PPV_ARGS(&pEvent));
		if (FAILED(hr))
			return hr;

		hr = pGraph->QueryInterface(IID_PPV_ARGS(&pEventSink));
		if (FAILED(hr))
			return hr;

		// Create the Sample Grabber filter.
		hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pGrabberF));
		if (FAILED(hr))
			return hr;

		hr = pGraph->AddFilter(pGrabberF, L"Sample Grabber");
		if (FAILED(hr))
			return hr;

		hr = pGrabberF->QueryInterface(IID_ISampleGrabber, (void**)&pGrabber);
		if (FAILED(hr))
			return hr;

		// Configure the media type output
		AM_MEDIA_TYPE mt;
		ZeroMemory(&mt, sizeof(mt));
		mt.majortype = MEDIATYPE_Video;
		mt.subtype = MEDIASUBTYPE_RGB32;

		hr = pGrabber->SetMediaType(&mt);
		if (FAILED(hr))
			return hr;

		// Find system device enumerator to find a video capture device.
		hr = CoCreateInstance(CLSID_SystemDeviceEnum, 0, CLSCTX_INPROC_SERVER, IID_ICreateDevEnum, (LPVOID*)&pDevEnum);
		hr = pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnumMoniker, 0);
		if (S_OK == pEnumMoniker->Next(1, &pMoniker, 0))
			pMoniker->BindToObject(0, 0, IID_IBaseFilter, (void **)&pSourceF);
		// Add sample grabber to the graph
		pGraph->AddFilter(pSourceF, L"Source");
		pDevEnum->Release();
		pEnumMoniker->Release();
		pMoniker->Release();



		hr = ConnectFilters(pGraph, pSourceF, pGrabberF);
		if (FAILED(hr))
			return hr;

		hr = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pNullF));
		if (FAILED(hr))
			return hr;

		hr = pGraph->AddFilter(pNullF, L"Null Filter");
		if (FAILED(hr))
			return hr;

		hr = ConnectFilters(pGraph, pGrabberF, pNullF);
		if (FAILED(hr))
			return hr;

		// After the first sample, the Sample Grabber continues to process samples. This is the default behavior.
		hr = pGrabber->SetOneShot(FALSE);
		if (FAILED(hr))
			return hr;

		// Set modes
		hr = pGrabber->SetBufferSamples(FALSE); // Buffer seems to be no use in callback mode
		if (FAILED(hr))
			return hr;

		// Set callback, 1 = BufferCB
		hr = pGrabber->SetCallback(&adapter, 1);
		if (FAILED(hr))
			return hr;

		// Necessary interfaces for controlling
		hr = pGraph->QueryInterface(IID_IMediaControl, (void**)(&pControl));
		if (FAILED(hr)) return hr;
		hr = pGraph->QueryInterface(IID_IMediaEventEx, (void**)(&pEvent));
		if (FAILED(hr)) return hr;
		hr = pGraph->QueryInterface(IID_IMediaSeeking, (void**)(&pSeeking));
		if (FAILED(hr)) return hr;

#ifdef DEBUG
		LONGLONG duration;
		double rate;
		pSeeking->GetRate(&rate);
		pSeeking->GetDuration(&duration);
		printf("Rate is %lf, duration is %lld", rate, duration);
#endif

		// Turn off the sync clock for max speed
		IMediaFilter *pMediaFilter = NULL;
		hr = pGraph->QueryInterface(IID_IMediaFilter, (void**)(&pMediaFilter));
		if (FAILED(hr)) return hr;
		hr = pMediaFilter->SetSyncSource(NULL);
		if (FAILED(hr)) return hr;
		SafeRelease(&pMediaFilter);

		// Query the media information
		pGrabber->GetConnectedMediaType(&mt);
		if ((mt.formattype == FORMAT_VideoInfo) &&
			(mt.cbFormat >= sizeof(VIDEOINFOHEADER)) &&
			(mt.pbFormat != NULL))
		{
			VIDEOINFOHEADER *pVih = (VIDEOINFOHEADER*)mt.pbFormat;
			this->_Width = pVih->bmiHeader.biWidth;
			this->_Height = pVih->bmiHeader.biHeight;
			this->_BitsPerPixel = pVih->bmiHeader.biBitCount;
		}
		else
			return -1;

		return S_OK;
	}

	void AviReader::ReleaseAll()
	{
		SafeRelease(&pMoniker);
		SafeRelease(&pEnumMoniker);
		SafeRelease(&pDevEnum);
		SafeRelease(&pSeeking);
		SafeRelease(&pPin);
		SafeRelease(&pEnum);
		SafeRelease(&pNullF);
		SafeRelease(&pSourceF);
		SafeRelease(&pGrabber);
		SafeRelease(&pGrabberF);
		SafeRelease(&pControl);
		SafeRelease(&pEventSink);
		SafeRelease(&pEvent);
		SafeRelease(&pGraph);
	}

	void AviReader::ProcessThread()
	{
		CoInitialize(NULL);
		Finished = FAILED(pszVideoFile != nullptr ? Init() : InitWebcam());
		SetEvent(hSyncResult);			// Init and report readiness
		WaitForSingleObject(hSyncQuery, INFINITE);
#ifdef DEBUG
		printf("AviReader::ProcessThread - begin frame processing\n");
#endif

		if (!Finished)
		{
			pControl->Run();				// Run the graph

			long evCode;
			LONG_PTR param1, param2;
			HRESULT hr;
			while (hr = pEvent->GetEvent(&evCode, &param1, &param2, INFINITE), SUCCEEDED(hr))
			{
				if (evCode == EC_COMPLETE || evCode == EC_USERABORT)
					break;

				hr = pEvent->FreeEventParams(evCode, param1, param2);
			}

			Finished = true;
			SetEvent(hSyncResult);
		}

		ReleaseAll();
		CoUninitialize();
	}

	void AviReader::OnBufferCB(double SampleTime, BYTE *pBuffer, long nBufferLen)
	{
		// If we are finished => quit!
		if (Finished)
		{
			pEventSink->Notify(EC_USERABORT, NULL, NULL);
			pControl->Stop();
			return;
		}

		this->pBuffer = pBuffer;

		AM_MEDIA_TYPE mt;
		pGrabber->GetConnectedMediaType(&mt);

		// Examine the format block.
		if ((mt.formattype == FORMAT_VideoInfo) &&
			(mt.cbFormat >= sizeof(VIDEOINFOHEADER)) &&
			(mt.pbFormat != NULL))
		{
			VIDEOINFOHEADER *pVih = (VIDEOINFOHEADER*)mt.pbFormat;
			check(_Width == pVih->bmiHeader.biWidth);
			check(_Height == pVih->bmiHeader.biHeight);
			check(_BitsPerPixel == pVih->bmiHeader.biBitCount);
		}
		else
		{
			Finished = true;
		}

		FreeMediaType(mt);

		// Notify that the data has been received
		SetEvent(hSyncResult);

#ifdef DEBUG
		printf("AviReader::OnBufferCB - new frame is ready\n");
#endif

		// Wait for data processed
		WaitForSingleObject(hSyncQuery, INFINITE);

#ifdef DEBUG
		printf("AviReader::OnBufferCB - waiting for new frame\n");
#endif
	}

	DWORD AviReader::_ProcessThread(LPVOID param)
	{
		AviReader* ar = (AviReader*)param;
		ar->ProcessThread();
		return 0;
	}

	AviReader::AviReader() :
		adapter(this)
	{

	}

	AviReader* AviReader::Create(PCWSTR pszVideoFile)
	{
		AviReader* ar = new AviReader();
		ar->hSyncResult = CreateEvent(NULL, false, false, NULL);
		ar->hSyncQuery = CreateEvent(NULL, false, false, NULL);
		ar->pszVideoFile = pszVideoFile;
		ar->hThread = CreateThread(NULL, 0, _ProcessThread, ar, 0, NULL);
		WaitForSingleObject(ar->hSyncResult, INFINITE);						// Wait for thread initialization
		SetEvent(ar->hSyncQuery);
		if (ar->Finished) { delete ar; return nullptr; }					// Failed to initialize => quit
		return ar;
	}

	AviReader* AviReader::CreateWebcam()
	{
		AviReader* ar = new AviReader();
		ar->hSyncResult = CreateEvent(NULL, false, false, NULL);
		ar->hSyncQuery = CreateEvent(NULL, false, false, NULL);
		ar->pszVideoFile = nullptr;
		ar->hThread = CreateThread(NULL, 0, _ProcessThread, ar, 0, NULL);
		WaitForSingleObject(ar->hSyncResult, INFINITE);						// Wait for thread initialization
		SetEvent(ar->hSyncQuery);
		if (ar->Finished) { delete ar; return nullptr; }					// Failed to initialize => quit
		return ar;
	}

	AviReader::~AviReader()
	{
		// The thread must be stopped here - notify it 
		Finished = true;
		SetEvent(hSyncQuery);

		// Wait for thread completion
		WaitForSingleObject(hThread, INFINITE);

		// Release all other handles
		CloseHandle(hThread);
		CloseHandle(hSyncResult);
		CloseHandle(hSyncQuery);
	}

	int AviReader::Width()
	{
		return _Width;
	}

	int AviReader::Height()
	{
		return _Height;
	}

	bool AviReader::GrabNextFrame(AviFrameData *data)
	{
		WaitForSingleObject(hSyncResult, INFINITE);
		if (Finished)
			return false;

		data->buffer = (BYTE*)pBuffer;
		data->Width = _Width;
		data->Height = _Height;

		return true;
	}

	void AviReader::Navigate(double Position)
	{
		LONGLONG Duration;
		LONGLONG Current = (LONGLONG)(Position * 10000000.0);
		pControl->Pause();
		pSeeking->GetDuration(&Duration);
		HRESULT hr = pSeeking->SetPositions(&Current, AM_SEEKING_AbsolutePositioning, NULL, AM_SEEKING_NoPositioning);
		ResetEvent(hSyncResult);
		SetEvent(hSyncQuery);
		hr = pControl->Run();
	}

	CustomBitmapImage<PixelByteRGBA>* AviReader::GrabNextFrame()
	{
		WaitForSingleObject(hSyncResult, INFINITE);
		if (Finished)
		{
#ifdef DEBUG
			printf("AviReader::GrabNextFrame -> nullptr\n");
#endif
			return nullptr;
		}

#ifdef DEBUG
		printf("AviReader::GrabNextFrame -> BufferImage\n");
#endif

		BufferImage.Init((byte*)pBuffer + (_Height - 1) * _Width * 4, _Width, _Height, -_Width * 4);
		return &BufferImage;
	}

	void AviReader::ContinueGrabbing()
	{
#ifdef DEBUG
		printf("AviReader::ContinueGrabbing\n");
#endif

		SetEvent(hSyncQuery);
	}
}

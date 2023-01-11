#pragma once

#include <Windows.h>
#include <dshow.h>
#include <evcode.h>

#include "../core.h"

//using namespace ip;
//using namespace ip::internal;

#pragma region From qedit.h

interface ISampleGrabberCB :
	public IUnknown
{
	virtual STDMETHODIMP SampleCB(double SampleTime, IMediaSample *pSample) = 0;
	virtual STDMETHODIMP BufferCB(double SampleTime, BYTE *pBuffer, long BufferLen) = 0;
};

static const IID IID_ISampleGrabberCB = { 0x0579154A, 0x2B53, 0x4994, { 0xB0, 0xD0, 0xE7, 0x73, 0x14, 0x8E, 0xFF, 0x85 } };

interface ISampleGrabber :
	public IUnknown
{
	virtual HRESULT STDMETHODCALLTYPE SetOneShot(BOOL OneShot) = 0;
	virtual HRESULT STDMETHODCALLTYPE SetMediaType(const AM_MEDIA_TYPE *pType) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetConnectedMediaType(AM_MEDIA_TYPE *pType) = 0;
	virtual HRESULT STDMETHODCALLTYPE SetBufferSamples(BOOL BufferThem) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetCurrentBuffer(long *pBufferSize, long *pBuffer) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetCurrentSample(IMediaSample **ppSample) = 0;
	virtual HRESULT STDMETHODCALLTYPE SetCallback(ISampleGrabberCB *pCallback, long WhichMethodToCallback) = 0;
};

static const IID IID_ISampleGrabber = { 0x6B652FFF, 0x11FE, 0x4fce, { 0x92, 0xAD, 0x02, 0x66, 0xB5, 0xD7, 0xC7, 0x8F } };
static const CLSID CLSID_SampleGrabber = { 0xC1F400A0, 0x3F08, 0x11d3, { 0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37 } };
static const CLSID CLSID_NullRenderer = { 0xC1F400A4, 0x3F08, 0x11d3, { 0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37 } };
static const CLSID CLSID_VideoEffects1Category = { 0xcc7bfb42, 0xf175, 0x11d1, { 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };
static const CLSID CLSID_VideoEffects2Category = { 0xcc7bfb43, 0xf175, 0x11d1, { 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };
static const CLSID CLSID_AudioEffects1Category = { 0xcc7bfb44, 0xf175, 0x11d1, { 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };
static const CLSID CLSID_AudioEffects2Category = { 0xcc7bfb45, 0xf175, 0x11d1, { 0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59 } };

#pragma endregion

namespace ip
{
	// Forward declaration of AviReader class
	class AviReader;

#pragma region Wrapper

	namespace internal
	{
		// SVideoProcessAdapter is only for providing a callback for ISampleGrabber.
		class SVideoProcessAdapter :
			public ISampleGrabberCB
		{
		public:
			SVideoProcessAdapter(AviReader *pAviReader);
			virtual ~SVideoProcessAdapter(void);

			STDMETHOD(BufferCB)(double SampleTime, BYTE *pBuffer, long nBufferLen);
			STDMETHOD(SampleCB)(double, IMediaSample *);
			STDMETHOD(QueryInterface)(REFIID iid, LPVOID *ppv);
			STDMETHOD_(ULONG, AddRef)();
			STDMETHOD_(ULONG, Release)();

		protected:
			AviReader *pAviReader;
		};
	}

#pragma endregion

#pragma pack(push, 1)

	struct AviFrameData
	{
		BYTE *buffer;
		int Width, Height;
	};

#pragma pack(pop)

	class AviReader
	{
		friend class internal::SVideoProcessAdapter;

	protected:
		PCWSTR pszVideoFile;
		IGraphBuilder *pGraph;
		IMediaControl *pControl;
		IMediaEventEx *pEvent;
		IMediaEventSink *pEventSink;
		IBaseFilter *pGrabberF;
		ISampleGrabber *pGrabber;
		IBaseFilter *pSourceF;
		IEnumPins *pEnum;
		IPin *pPin;
		IBaseFilter *pNullF;
		IMediaSeeking *pSeeking;

		ICreateDevEnum *pDevEnum;
		IEnumMoniker *pEnumMoniker;
		IMoniker *pMoniker;

		internal::SVideoProcessAdapter adapter;

		long cbBuffer;
		volatile BYTE *pBuffer;
		CustomBitmapImage<PixelByteRGBA> BufferImage;

		int _Width, _Height, _BitsPerPixel;

		// MultiThread
		HANDLE hThread;
		HANDLE hSyncResult, hSyncQuery;
		volatile bool Finished;

		// Util methods

		template <class T> void SafeRelease(T **ppT)
		{
			if (*ppT)
			{
				(*ppT)->Release();
				*ppT = NULL;
			}
		}

		HRESULT	GetPin(IBaseFilter *pFilter, PIN_DIRECTION PinDir, IPin **ppPin);
		HRESULT ConnectFilters(IGraphBuilder *pGraph, IBaseFilter *pFirst, IBaseFilter *pSecond);
		void FreeMediaType(AM_MEDIA_TYPE& mt);

		// ProcessThread

		HRESULT Init();
		HRESULT InitWebcam();
		void ReleaseAll();
		void ProcessThread();
		void OnBufferCB(double SampleTime, BYTE *pBuffer, long nBufferLen);
		static DWORD __stdcall _ProcessThread(LPVOID param);

		// Main section

		AviReader();

	public:

		static AviReader* CreateWebcam();
		static AviReader* Create(PCWSTR pszVideoFile);
		~AviReader();

		int Width();
		int Height();
		void Navigate(double Position);

		bool GrabNextFrame(AviFrameData *data);
		CustomBitmapImage<PixelByteRGBA>* GrabNextFrame();
		void ContinueGrabbing();
	};
}
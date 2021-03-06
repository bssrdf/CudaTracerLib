#pragma once

#include <Kernel/Tracer.h>
#include <Base/CudaMemoryManager.h>
#include <Kernel/DoubleRayBuffer.h>
#include <Math/half.h>
#include <Kernel/BlockSampler/BlockSamplerBuffer.h>

namespace CudaTracerLib {

struct WavefrontPTRayData
{
	Spectrum throughput;
	half x, y;
	Spectrum L;
	Spectrum directF;
	float dDist;
	unsigned int dIdx;
	bool specular_bounce;
	float bsdf_pdf;
	unsigned int prev_normal;
};

typedef DoubleRayBuffer<WavefrontPTRayData> WavefrontPathTracerBuffer;

class WavefrontPathTracer : public Tracer<true>, public IDepthTracer
{
public:
	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(int, MaxPathLength)
	PARAMETER_KEY(int, RRStartDepth)

	WavefrontPathTracer()
		: m_ray_buf(0)
	{
		m_sParameters << KEY_Direct()				<< CreateSetBool(true)
					  << KEY_MaxPathLength()		<< CreateInterval<int>(50, 1, INT_MAX)
					  << KEY_RRStartDepth()			<< CreateInterval(5, 1, INT_MAX);
	}
	~WavefrontPathTracer()
	{
		if (m_ray_buf)
		{
			m_ray_buf->Free();
			delete m_ray_buf;
		}

		m_blockBuffer.Free();
	}
	virtual void Resize(unsigned int w, unsigned int h)
	{
		Tracer<true>::Resize(w, h);

		if (m_ray_buf)
		{
			m_ray_buf->Free();
			delete m_ray_buf;
		}
		m_ray_buf = new WavefrontPathTracerBuffer(w * h, w * h);

		m_blockBuffer.Resize(w, h);
	}
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
private:
	WavefrontPathTracerBuffer* m_ray_buf;
	BlockSamplerBuffer m_blockBuffer;
};

}
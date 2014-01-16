#pragma once

#include <MathTypes.h>
#include "..\Engine\e_Mesh.h"
#include "..\Math\Sampling.h"
#include "e_Samples.h"

#define MAX_SHAPE_LENGTH 32

struct CUDA_ALIGN(16) ShapeSet
{
	CUDA_ALIGN(16) struct triData
	{
		CUDA_ALIGN(16) float3 p[3];
		CUDA_ALIGN(16) float3 n;
		CUDA_ALIGN(16) float area;
		CUDA_ALIGN(16) e_StreamReference(e_TriIntersectorData) datRef;

		CUDA_FUNC_IN void UniformSampleTriangle(float u1, float u2, float *u, float *v) const
		{
			float su1 = sqrtf(u1);
			*u = 1.f - su1;
			*v = u2 * su1;
		}
		
		triData(){}
		triData(e_StreamReference(e_TriIntersectorData) ref, float4x4& mat)
		{
			datRef = ref;
			Recalculate(mat);
		}
		CUDA_FUNC_IN float3 rndPoint(float b1, float b2) const
		{
			return b1 * p[0] + b2 * p[1] + (1.f - b1 - b2) * p[2];
		}
		CUDA_DEVICE CUDA_HOST float3 Sample(float u1, float u2, float3* Ns = 0, float* a = 0, float* b = 0) const;
		CUDA_FUNC_IN float3 nor() const
		{
			return n;
		}
		AABB box() const;
		void Recalculate(const float4x4& mat);
	};
public:
	ShapeSet(){}
    ShapeSet(e_StreamReference(e_TriIntersectorData)* indices, unsigned int indexCount, float4x4& mat);
    CUDA_FUNC_IN float Area() const { return sumArea; }
	CUDA_DEVICE CUDA_HOST void SamplePosition(PositionSamplingRecord& pRec, const float2& spatialSample) const;
    CUDA_FUNC_IN float Pdf(const PositionSamplingRecord &p) const
	{
		return 1.0f / sumArea;
	}
	AABB getBox() const
	{
		AABB b = AABB::Identity();
		for(int i = 0; i < count; i++)
			b.Enlarge(tris[i].box());
		return b;
	}
	void Recalculate(const float4x4& mat);
private:
    CUDA_ALIGN(16) triData tris[MAX_SHAPE_LENGTH];
    float sumArea;
    CUDA_ALIGN(16) Distribution1D<MAX_SHAPE_LENGTH> areaDistribution;
	int count;
};
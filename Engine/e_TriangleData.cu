#include "e_TriangleData.h"
#include "e_Node.h"
#include "..\Math\Compression.h"

#ifdef EXT_TRI

e_TriangleData::e_TriangleData(float3* P, unsigned char matIndex, float2* T, float3* N, float3* Tan, float3* BiTan)
{
	//unsigned short s = NormalizedFloat3ToUchar2(make_float3(0,0,1)), t = NormalizedFloat3ToUchar2(make_float3(0,0,-1));
	//float3 a = Uchar2ToNormalizedFloat3(s), b = Uchar2ToNormalizedFloat3(t);

	m_sHostData.MatIndex = matIndex;
	for(int i = 0; i < 3; i++)
	{
		m_sHostData.Normals[i] = NormalizedFloat3ToUchar2(normalize(N[i]));
		m_sHostData.Tangents[i] = NormalizedFloat3ToUchar2(normalize(Tan[i]));
		half2 h2 = half2(T[i]);
		m_sHostData.TexCoord[i] = *(ushort2*)&h2;
		//NOR[i] = N[i];
	}
}

void e_TriangleData::setData(const float3& na, const float3& nb, const float3& nc,
									   const float3& ta, const float3& tb, const float3& tc)
{
	m_sDeviceData.Row0.x = NormalizedFloat3ToUchar2(na) | (NormalizedFloat3ToUchar2(nb) << 16);
	m_sDeviceData.Row0.y = NormalizedFloat3ToUchar2(nc) | (NormalizedFloat3ToUchar2(ta) << 16);
	m_sDeviceData.Row0.z = NormalizedFloat3ToUchar2(tb) | (NormalizedFloat3ToUchar2(tc) << 16);
}

void e_TriangleData::lerpFrame(const float2& bCoords, const float4x4& localToWorld, Frame& sys, float3* ng) const
{
	//float3 na = NOR[0], nb = NOR[1], nc = NOR[2];
	uint4 q = m_sDeviceData.Row0;
	float3 na = Uchar2ToNormalizedFloat3(q.x), nb = Uchar2ToNormalizedFloat3(q.x >> 16), nc = Uchar2ToNormalizedFloat3(q.y);
	float3 ta = Uchar2ToNormalizedFloat3(q.y >> 16), tb = Uchar2ToNormalizedFloat3(q.z), tc = Uchar2ToNormalizedFloat3(q.z >> 16);
	float w = 1.0f - bCoords.x - bCoords.y, u = bCoords.x, v = bCoords.y;
	sys.n = (u * na + v * nb + w * nc);
	sys.t = (u * ta + v * tb + w * tc);

	sys = sys * localToWorld;
	sys.n = normalize(sys.n);
	sys.s = normalize(cross(sys.t, sys.n));
	sys.t = normalize(cross(sys.s, sys.n));
	if(ng)
		*ng = normalize(localToWorld.TransformNormal((na + nb + nc) / 3.0f));//that aint the def of a face normal but to get the vertices we would have to invert the matrix
}

float2 e_TriangleData::lerpUV(const float2& bCoords) const
{
#ifdef ISCUDA
	#define tof2(x) make_float2(__half2float(x & 0xffff), __half2float(x >> 16))
#else
	#define tof2(x) make_float2(half((unsigned short)(x & 0xffff)).ToFloat(), half((unsigned short)(x & 0xffff0000)).ToFloat())
#endif
	float2 a = tof2(m_sDeviceData.Row1.x), b = tof2(m_sDeviceData.Row1.y), c = tof2(m_sDeviceData.Row1.z);
	float u = bCoords.y, v = 1.0f - u - bCoords.x;
	return a + u * (b - a) + v * (c - a);
#undef tof2
}

void e_TriangleData::getNormalDerivative(const float2& bCoords, float3& dndu, float3& dndv) const
{
	uint4 q = m_sDeviceData.Row0;
	float3 n0 = Uchar2ToNormalizedFloat3(q.x), n1 = Uchar2ToNormalizedFloat3(q.x >> 16), n2 = Uchar2ToNormalizedFloat3(q.y);
#ifdef ISCUDA
	#define tof2(x) make_float2(__half2float(x & 0xffff), __half2float(x >> 16))
#else
	#define tof2(x) make_float2(half((unsigned short)(x & 0xffff)).ToFloat(), half((unsigned short)(x & 0xffff0000)).ToFloat())
#endif
	float2 uv0 = tof2(m_sDeviceData.Row1.x), uv1 = tof2(m_sDeviceData.Row1.y), uv2 = tof2(m_sDeviceData.Row1.z);
	float w = 1.0f - bCoords.x - bCoords.y, u = bCoords.x, v = bCoords.y;

	float3 N = u * n1 + v * n2 + w * n0;
	float il = 1.0f / length(N);
	N *= il;
	dndu = (n1 - n0) * il; dndu -= N * dot(N, dndu);
	dndv = (n2 - n0) * il; dndv -= N * dot(N, dndv);

	float2 duv1 = uv1 - uv0, duv2 = uv2 - uv0;
	float det = duv1.x * duv2.y - duv1.y * duv2.x;
	float invDet = 1.0f / det;
	float3 dndu_ = ( duv2.y * dndu - duv1.y * dndv) * invDet;
	float3 dndv_ = (-duv2.x * dndu + duv1.x * dndv) * invDet;
	dndu = dndu_; dndv = dndv_;
}
#endif

bool TraceResult::hasHit() const
{
	return m_pTri != 0;
}

TraceResult::operator bool() const
{
	return hasHit();
}

void TraceResult::Init()
{
	m_fDist = FLT_MAX;
	m_pNode = 0;
	m_pTri = 0;
}

unsigned int TraceResult::getMatIndex() const
{
	return m_pTri->getMatIndex(m_pNode->m_uMaterialOffset);
}

float2 TraceResult::lerpUV() const
{
	return m_pTri->lerpUV(m_fUV);
}

void TraceResult::getBsdfSample(const Ray& r, CudaRNG& _rng, BSDFSamplingRecord* bRec, const float3& wo) const
{
	getBsdfSample(r, _rng, bRec);
	bRec->wo = bRec->map.sys.toLocal(wo);
}
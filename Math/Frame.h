#pragma once

#include "float4x4.h"

//Implementation copied from Mitsuba. Slight additions with respect to matrices.

namespace CudaTracerLib {

CUDA_FUNC_IN void coordinateSystem(const NormalizedT<Vec3f>& a, NormalizedT<Vec3f>& s, NormalizedT<Vec3f>& t)
{
	if (math::abs(a.x) > math::abs(a.y))
	{
		float invLen = 1.0f / math::sqrt(a.x * a.x + a.z * a.z);
		t = NormalizedT<Vec3f>(a.z * invLen, 0.0f, -a.x * invLen);
	}
	else
	{
		float invLen = 1.0f / math::sqrt(a.y * a.y + a.z * a.z);
		t = NormalizedT<Vec3f>(0.0f, a.z * invLen, -a.y * invLen);
	}
	s = normalize(cross(t, a));
}

struct Frame
{
	NormalizedT<Vec3f> s, t;
	NormalizedT<Vec3f> n;
	CUDA_FUNC_IN Frame() { }
	CUDA_FUNC_IN Frame(const NormalizedT<Vec3f> &s, const NormalizedT<Vec3f> &t, const NormalizedT<Vec3f> &n)
		: s(s), t(t), n(n) {
	}
	CUDA_FUNC_IN Frame(const NormalizedT<Vec3f> &n) : n(n) {
		coordinateSystem(n, s, t);
	}

	CUDA_FUNC_IN Vec3f toLocal(const Vec3f &v) const {
		return Vec3f(dot(v, s), dot(v, t), dot(v, n));
	}
	CUDA_FUNC_IN Vec3f toWorld(const Vec3f &v) const {
		return s * v.x + t * v.y + n * v.z;
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> toLocal(const NormalizedT<Vec3f> &v) const {
		return NormalizedT<Vec3f>(toLocal((Vec3f)v));
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> toWorld(const NormalizedT<Vec3f> &v) const {
		return NormalizedT<Vec3f>(toWorld((Vec3f)v));
	}

	CUDA_FUNC_IN float4x4 ToWorldMatrix() const
	{
		float4x4 r;
		r.col(0, Vec4f(s, 0));
		r.col(1, Vec4f(t, 0));
		r.col(2, Vec4f(n, 0));
		r.col(3, Vec4f(0, 0, 0, 1));
		return r;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared cosine of the angle between the float3 and v */
	CUDA_FUNC_IN static float cosTheta2(const NormalizedT<Vec3f> &v) {
		return v.z * v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the cosine of the angle between the float3 and v */
	CUDA_FUNC_IN static float cosTheta(const NormalizedT<Vec3f> &v) {
		return v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared sine of the angle between the float3 and v */
	CUDA_FUNC_IN static float sinTheta2(const NormalizedT<Vec3f> &v) {
		return 1.0f - v.z * v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the sine of the angle between the float3 and v */
	CUDA_FUNC_IN static float sinTheta(const NormalizedT<Vec3f> &v) {
		float temp = sinTheta2(v);
		if (temp <= 0.0f)
			return 0.0f;
		return math::sqrt(temp);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the tangent of the angle between the float3 and v */
	CUDA_FUNC_IN static float tanTheta(const NormalizedT<Vec3f> &v) {
		float temp = 1 - v.z*v.z;
		if (temp <= 0.0f)
			return 0.0f;
		return math::sqrt(temp) / v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared tangent of the angle between the float3 and v */
	CUDA_FUNC_IN static float tanTheta2(const NormalizedT<Vec3f> &v) {
		float temp = 1 - v.z*v.z;
		if (temp <= 0.0f)
			return 0.0f;
		return temp / (v.z * v.z);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the sine of the phi parameter in spherical coordinates */
	CUDA_FUNC_IN static float sinPhi(const NormalizedT<Vec3f> &v) {
		float sinTheta = Frame::sinTheta(v);
		if (sinTheta == 0.0f)
			return 1.0f;
		return math::clamp(v.y / sinTheta, -1.0f, 1.0f);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the cosine of the phi parameter in spherical coordinates */
	CUDA_FUNC_IN static float cosPhi(const NormalizedT<Vec3f> &v) {
		float sinTheta = Frame::sinTheta(v);
		if (sinTheta == 0.0f)
			return 1.0f;
		return math::clamp(v.x / sinTheta, -1.0f, 1.0f);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared sine of the phi parameter in  spherical
	 * coordinates */
	CUDA_FUNC_IN static float sinPhi2(const NormalizedT<Vec3f> &v) {
		return math::clamp(v.y * v.y / sinTheta2(v), 0.0f, 1.0f);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared cosine of the phi parameter in  spherical
	 * coordinates */
	CUDA_FUNC_IN static float cosPhi2(const NormalizedT<Vec3f> &v) {
		return math::clamp(v.x * v.x / sinTheta2(v), 0.0f, 1.0f);
	}

	/// Reflection in local coordinates
	CUDA_FUNC_IN static Vec3f reflect(const Vec3f &wi) {
		return Vec3f(-wi.x, -wi.y, wi.z);
	}

	CUDA_FUNC_IN static NormalizedT<Vec3f> reflect(const NormalizedT<Vec3f> &wi)
	{
		return NormalizedT<Vec3f>(reflect((Vec3f)wi));
	}

	/// Refraction in local coordinates
	CUDA_FUNC_IN static Vec3f refract(const Vec3f &wi, float cosThetaT, float eta, float invEta) {
		float scale = -(cosThetaT < 0 ? invEta : eta);
		return Vec3f(scale*wi.x, scale*wi.y, cosThetaT);
	}

	CUDA_FUNC_IN static NormalizedT<Vec3f> refract(const NormalizedT<Vec3f> &wi, float cosThetaT, float eta, float invEta)
	{
		return NormalizedT<Vec3f>(refract((Vec3f)wi, cosThetaT, eta, invEta).normalized());
	}

	CUDA_FUNC_IN Frame operator *(const float4x4& m) const
	{
		Frame r;
		r.s = normalize(m.TransformDirection(s));
		r.t = normalize(m.TransformDirection(t));
		r.n = normalize(cross(r.t, r.s));
		return r;
	}

	CUDA_FUNC_IN bool operator==(const Frame &frame) const {
		Vec3f diff = frame.s - s + frame.t - t + frame.n - n;
		return !dot(diff, diff);
	}

	CUDA_FUNC_IN bool operator!=(const Frame &frame) const {
		return !operator==(frame);
	}

	friend std::ostream& operator<< (std::ostream & os, const Frame& rhs)
	{
		os << "N = " << rhs.n << "\nT = " << rhs.t << "\nB = " << rhs.s;
		return os;
	}
};

}
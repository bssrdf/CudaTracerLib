#pragma once

#include "MathFunc.h"
#include "Vector.h"
#include "float4x4.h"

namespace CudaTracerLib {

class Quaternion
{
public:
	Vec4f val;
	inline float& operator[](int n) { return *(((float*)&val) + n); }
	inline float operator[](int n) const { return *(((float*)&val) + n); }
	CUDA_FUNC_IN Quaternion(){}
	CUDA_FUNC_IN Quaternion(float x, float y, float z, float w)
		: val(x, y, z, w)
	{
	}
	CUDA_FUNC_IN Quaternion(float x, float y, float z)
	{
		float w = 1.0f - x*x - y*y - z*z;
		w = w < 0.0 ? 0.0f : (float)-sqrt(double(w));
		val = Vec4f(x, y, z, w);
		normalize();
	}
	//http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
	CUDA_FUNC_IN Quaternion(const float4x4& mat)
	{
		float trace = mat(0, 0) + mat(1, 1) + mat(2, 2); // I removed + 1.0f; see discussion with Ethan
		if (trace > 0)
		{// I changed M_EPSILON to 0
			float s = 0.5f / math::sqrt(trace + 1.0f);
			val.w = 0.25f / s;
			val.x = (mat(2, 1) - mat(1, 2)) * s;
			val.y = (mat(0, 2) - mat(2, 0)) * s;
			val.z = (mat(1, 0) - mat(0, 1)) * s;
		}
		else
		{
			if (mat(0, 0) > mat(1, 1) && mat(0, 0) > mat(2, 2))
			{
				float s = 2.0f * math::sqrt(1.0f + mat(0, 0) - mat(1, 1) - mat(2, 2));
				val.w = (mat(2, 1) - mat(1, 2)) / s;
				val.x = 0.25f * s;
				val.y = (mat(0, 1) + mat(1, 0)) / s;
				val.z = (mat(0, 2) + mat(2, 0)) / s;
			}
			else if (mat(1, 1) > mat(2, 2))
			{
				float s = 2.0f * math::sqrt(1.0f + mat(1, 1) - mat(0, 0) - mat(2, 2));
				val.w = (mat(0, 2) - mat(2, 0)) / s;
				val.x = (mat(0, 1) + mat(1, 0)) / s;
				val.y = 0.25f * s;
				val.z = (mat(1, 2) + mat(2, 1)) / s;
			}
			else
			{
				float s = 2.0f * math::sqrt(1.0f + mat(2, 2) - mat(0, 0) - mat(1, 1));
				val.w = (mat(1, 0) - mat(0, 1)) / s;
				val.x = (mat(0, 2) + mat(2, 0)) / s;
				val.y = (mat(1, 2) + mat(2, 1)) / s;
				val.z = 0.25f * s;
			}
		}
	}
	CUDA_FUNC_IN Quaternion operator *(const Quaternion &q) const
	{
		Quaternion r;
		r.val.w = val.w*q.val.w - val.x*q.val.x - val.y*q.val.y - val.z*q.val.z;
		r.val.x = val.w*q.val.x + val.x*q.val.w + val.y*q.val.z - val.z*q.val.y;
		r.val.y = val.w*q.val.y + val.y*q.val.w + val.z*q.val.x - val.x*q.val.z;
		r.val.z = val.w*q.val.z + val.z*q.val.w + val.x*q.val.y - val.y*q.val.x;
		return r;
	}
	CUDA_FUNC_IN Vec3f operator *(const Vec3f &v) const
	{
		float x = val.x + val.x;
		float y = val.y + val.y;
		float z = val.z + val.z;
		float wx = val.w * x;
		float wy = val.w * y;
		float wz = val.w * z;
		float xx = val.x * x;
		float xy = val.x * y;
		float xz = val.x * z;
		float yy = val.y * y;
		float yz = val.y * z;
		float zz = val.z * z;
		Vec3f vector;
		vector.x = ((v.x * ((1.0f - yy) - zz)) + (v.y * (xy - wz))) + (v.z * (xz + wy));
		vector.y = ((v.x * (xy + wz)) + (v.y * ((1.0f - xx) - zz))) + (v.z * (yz - wx));
		vector.z = ((v.x * (xz - wy)) + (v.y * (yz + wx))) + (v.z * ((1.0f - xx) - yy));
		return vector;
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> operator *(const NormalizedT<Vec3f> &v) const
	{
		return NormalizedT<Vec3f>(operator*((Vec3f)v));
	}
	CUDA_FUNC_IN const Quaternion & operator *= (const Quaternion &q)
	{
		val.w = val.w*q.val.w - val.x*q.val.x - val.y*q.val.y - val.z*q.val.z;
		val.x = val.w*q.val.x + val.x*q.val.w + val.y*q.val.z - val.z*q.val.y;
		val.y = val.w*q.val.y + val.y*q.val.w + val.z*q.val.x - val.x*q.val.z;
		val.z = val.w*q.val.z + val.z*q.val.w + val.x*q.val.y - val.y*q.val.x;
		return *this;
	}
	CUDA_FUNC_IN static Quaternion buildFromAxisAngle(const Vec3f& axis, float angle)
	{
		float radians = (angle / 180.0f)*3.14159f;

		// cache this, since it is used multiple times below
		float sinThetaDiv2 = (float)sin((radians / 2.0f));

		Quaternion ret;
		// now calculate the components of the quaternion
		ret.val.x = axis.x * sinThetaDiv2;
		ret.val.y = axis.y * sinThetaDiv2;
		ret.val.z = axis.z * sinThetaDiv2;

		ret.val.w = (float)cos((radians / 2.0f));
		return ret;
	}
	CUDA_FUNC_IN void normalize()
	{
		val = CudaTracerLib::normalize(val);
	}
	CUDA_FUNC_IN Quaternion conjugate() const { return Quaternion(-val.x, -val.y, -val.z, val.w); }
	CUDA_FUNC_IN float length() const
	{
		return CudaTracerLib::length(val);
	}
	CUDA_FUNC_IN Quaternion pow(float t) const
	{
		Quaternion result(0, 0, 0, 0);

		if (math::abs(val.w) < 0.9999)
		{
			float alpha = (float)acos(val.w);
			float newAlpha = alpha * t;

			result.val.w = (float)cos(newAlpha);
			float fact = float(sin(newAlpha) / sin(alpha));
			result.val.x *= fact;
			result.val.y *= fact;
			result.val.z *= fact;
		}
		return result;
	}
	CUDA_FUNC_IN NormalizedT<OrthogonalAffineMap> toMatrix() const
	{
		float xx = val.x * val.x;
		float yy = val.y * val.y;
		float zz = val.z * val.z;
		float xy = val.x * val.y;
		float zw = val.z * val.w;
		float zx = val.z * val.x;
		float yw = val.y * val.w;
		float yz = val.y * val.z;
		float xw = val.x * val.w;
		NormalizedT<OrthogonalAffineMap> r;
		r.col(0, Vec4f(1.0f - (2.0f * (yy + zz)), 2.0f * (xy + zw), 2.0f * (zx - yw), 0));
		r.col(1, Vec4f(2.0f * (xy - zw), 1.0f - (2.0f * (zz + xx)), 2.0f * (yz + xw), 0));
		r.col(2, Vec4f(2.0f * (zx + yw), 2.0f * (yz - xw), 1.0f - (2.0f * (yy + xx)), 0));
		r.col(3, Vec4f(0, 0, 0, 1));
		return r;
	}
	CUDA_FUNC_IN static Quaternion slerp(const Quaternion &q1, const Quaternion &q2, float t)
	{
		Quaternion result, _q2 = q2;

		float cosOmega = q1.val.w * q2.val.w + q1.val.x * q2.val.x + q1.val.y * q2.val.y + q1.val.z * q2.val.z;

		if (cosOmega < 0.0f)
		{
			_q2.val.x = -_q2.val.x;
			_q2.val.y = -_q2.val.y;
			_q2.val.z = -_q2.val.z;
			_q2.val.w = -_q2.val.w;
			cosOmega = -cosOmega;
		}

		float k0, k1;
		if (cosOmega > 0.99999f)
		{
			k0 = 1.0f - t;
			k1 = t;
		}
		else
		{
			float sinOmega = (float)sqrt(1.0f - cosOmega*cosOmega);
			float omega = (float)atan2(sinOmega, cosOmega);

			float invSinOmega = 1.0f / sinOmega;

			k0 = float(sin(((1.0f - t)*omega)))*invSinOmega;
			k1 = float(sin(t*omega))*invSinOmega;
		}
		result.val.x = q1.val.x * k0 + _q2.val.x * k1;
		result.val.y = q1.val.y * k0 + _q2.val.y * k1;
		result.val.z = q1.val.z * k0 + _q2.val.z * k1;
		result.val.w = q1.val.w * k0 + _q2.val.w * k1;

		return result;
	}

	friend std::ostream& operator<< (std::ostream & os, const Quaternion& rhs)
	{
		os << rhs.val;
		return os;
	}
};

}
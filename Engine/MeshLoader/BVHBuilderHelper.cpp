#include <StdAfx.h>
#include <Base/FileStream.h>
#include "BVHBuilderHelper.h"
#include <Engine/SpatialStructures/BVH/SplitBVHBuilder.hpp>

namespace CudaTracerLib {

namespace bvh_helper
{
class clb : public IBVHBuilderCallback
{
	const Vec3f* V;
	const unsigned int* Iab;
	unsigned int v_count, i_count;
	unsigned int _index(unsigned int i, unsigned int o) const
	{
		return Iab ? Iab[i * 3 + o] : (i * 3 + o);
	}
public:
	std::vector<BVHNodeData>& nodes;
	std::vector<TriIntersectorData>& tris;
	std::vector<TriIntersectorData2>& indices;
	AABB box;
	int startNode;
	//unsigned int L0, L1;
	unsigned int l0, l1;
public:
	clb(unsigned int _v, unsigned int _i, const Vec3f* _V, const unsigned int* _I, std::vector<BVHNodeData>& A, std::vector<TriIntersectorData>& B, std::vector<TriIntersectorData2>& C)
		: V(_V), Iab(_I), v_count(_v), i_count(_i), nodes(A), tris(B), indices(C), l0(0), l1(0)
	{
	}

	virtual void startConstruction(unsigned int nInnerNodes, unsigned int nLeafNodes)
	{
		nodes.resize(nInnerNodes + 2);
		tris.resize(nLeafNodes + 2);
		indices.resize(nLeafNodes + 2);
	}

	virtual void iterateObjects(std::function<void(unsigned int, const AABB&)> f)
	{
		for (unsigned int j = 0; j < i_count / 3; j++)
		{
			AABB out = AABB::Identity();
			for (int i = 0; i < 3; i++)
				out = out.Extend(V[_index(j, i)]);
			f(j, out);
		}
	}

	virtual unsigned int createLeafNode(unsigned int parentBVHNodeIdx, const std::vector<unsigned int>& objIndices)
	{
		unsigned int firstIdx = l1;
		l1 += (unsigned int)objIndices.size();
		for (size_t i = 0; i < objIndices.size(); i++)
		{
			indices[firstIdx + i].setFlag(i == objIndices.size() - 1);
			indices[firstIdx + i].setIndex(objIndices[i]);
			tris[firstIdx + i].setData(V[_index(objIndices[i], 0)], V[_index(objIndices[i], 1)], V[_index(objIndices[i], 2)]);
		}
		return firstIdx;
	}

	virtual void finishConstruction(unsigned int startNode, const AABB& sceneBox)
	{
		this->box = box;
		this->startNode = startNode;
	}

	virtual unsigned int createInnerNode(BVHNodeData*& innerNode)
	{
		if (l0 >= nodes.size())
			throw std::runtime_error(__FUNCTION__);
		innerNode = &nodes[l0];
		return l0++;
	}

	virtual bool SplitNode(unsigned int index, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		lBox = rBox = AABB::Identity();
		Vec3f v1 = V[_index(index, 2)];
		for (int i = 0; i < 3; i++)
		{
			Vec3f v0 = v1;
			v1 = V[_index(index, i)];
			float V0[] = { v0.x, v0.y, v0.z };
			float V1[] = { v1.x, v1.y, v1.z };
			float v0p = V0[dim];
			float v1p = V1[dim];

			// Insert vertex to the boxes it belongs to.

			if (v0p <= pos)
				lBox = lBox.Extend(v0);
			if (v0p >= pos)
				rBox = rBox.Extend(v0);

			// Edge intersects the plane => insert intersection to both boxes.

			if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
			{
				Vec3f t = math::lerp(v0, v1, math::clamp01((pos - v0p) / (v1p - v0p)));
				lBox = lBox.Extend(t);
				rBox = rBox.Extend(t);
			}
		}
		lBox.maxV[dim] = pos;
		rBox.minV[dim] = pos;
		lBox = lBox.Intersect(refBox);
		rBox = rBox.Intersect(refBox);
		return true;
	}
};
}

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, unsigned int vCount, unsigned int cCount, BVH_Construction_Result& out)
{
	bvh_helper::clb c(vCount, cCount, vertices, indices, out.nodes, out.tris, out.tris2);
	SplitBVHBuilder::Platform P; P.m_maxLeafSize = 8;
	SplitBVHBuilder bu(&c, P, SplitBVHBuilder::BuildParams());
	bu.run();
	BVH_Construction_Result r;
	r.box = c.box;
	r.tris2 = c.indices;
	r.nodes = c.nodes;
	r.tris = c.tris;
}

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, int vCount, int cCount, FileOutputStream& O, BVH_Construction_Result* out)
{
	BVH_Construction_Result localRes;
	if (!out)
		out = &localRes;

	bvh_helper::clb c(vCount, cCount, vertices, indices, out->nodes, out->tris, out->tris2);
	SplitBVHBuilder::Platform P; P.m_maxLeafSize = 8;
	SplitBVHBuilder bu(&c, P, SplitBVHBuilder::BuildParams()); bu.run();
	O << (unsigned long long)c.l0;
	if (c.l0)
		O.Write(&c.nodes[0], (unsigned int)c.l0 * sizeof(BVHNodeData));
	O << (unsigned long long)c.l1;
	if (c.l1)
		O.Write(&c.tris[0], (unsigned int)c.l1 * sizeof(TriIntersectorData));
	O << (unsigned long long)c.l1;
	if (c.l1)
		O.Write(&c.indices[0], (unsigned int)c.l1 * sizeof(TriIntersectorData2));
}

}
#pragma once

#include "e_Grid.h"
#include "../CudaMemoryManager.h"

template<typename T> struct e_SpatialLinkedMap_volume_iterator;

//a mapping from R^3 -> T^n, ie. associating variable number of values with each point in the grid
template<typename T> struct e_SpatialLinkedMap
{
	struct linkedEntry
	{
		unsigned int nextIdx;
		T value;
	};

	unsigned int numData, gridSize;
	linkedEntry* deviceData;
	unsigned int* deviceMap;
	unsigned int deviceDataIdx;
	k_HashGrid_Reg hashMap;
public:
	typedef e_SpatialLinkedMap_volume_iterator<T> iterator;

	CUDA_FUNC_IN e_SpatialLinkedMap(){}
	e_SpatialLinkedMap(unsigned int gridSize, unsigned int numData)
		: numData(numData), gridSize(gridSize)
	{
		CUDA_MALLOC(&deviceData, sizeof(linkedEntry) * numData);
		CUDA_MALLOC(&deviceMap, sizeof(unsigned int) * gridSize * gridSize * gridSize);
	}
	void Free()
	{
		CUDA_FREE(deviceData);
		CUDA_FREE(deviceMap);
	}

	void SetSceneDimensions(const AABB& box, float initialRadius)
	{
		hashMap = k_HashGrid_Reg(box, initialRadius, gridSize * gridSize * gridSize);
	}

	void ResetBuffer()
	{
		deviceDataIdx = 0;
		ThrowCudaErrors(cudaMemset(deviceMap, -1, sizeof(unsigned int) * gridSize * gridSize * gridSize));
	}

	CUDA_FUNC_IN bool isFull() const
	{
		return deviceDataIdx >= numData;
	}

	CUDA_FUNC_IN bool store(const Vec3f& p, const T& v)
	{
		//build linked list and spatial map
		unsigned int data_idx = Platform::Increment(&deviceDataIdx);
		if(deviceDataIdx >= numData)
			return false;
		unsigned int map_idx = hashMap.Hash(p);
		unsigned int old_idx = Platform::Exchange(deviceMap + map_idx, data_idx);
		//copy actual data
		deviceData[data_idx].value = v;
		deviceData[data_idx].nextIdx = old_idx;
		return true;
	}

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> begin(const Vec3f& p) const;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> end(const Vec3f& p) const;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> begin(const Vec3f& min, const Vec3f& max) const;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> end(const Vec3f& min, const Vec3f& max) const;

	template<typename CLB> CUDA_FUNC_IN void ForAll(const Vec3f& p, const CLB& clb)
	{
		auto e = end(p);
		for (auto it = begin(p); it != e; ++it)
			clb(it.getDataIdx(), *it);
	}

	template<typename CLB> CUDA_FUNC_IN void ForAll(const Vec3f& min, const Vec3f& max, const CLB& clb)
	{
		auto e = end(min, max);
		for (auto it = begin(min, max); it != e; ++it)
			clb(it.getDataIdx(), *it);
	}

	template<typename CLB> CUDA_FUNC_IN void ForAll(const Vec3u& p, const CLB& clb)
	{
		auto e = e_SpatialLinkedMap_volume_iterator<T>(*this, p, p, true);
		for (auto it = e_SpatialLinkedMap_volume_iterator<T>(*this, p, p, false); it != e; ++it)
			clb(it.getDataIdx(), *it);
	}

	template<typename CLB> CUDA_FUNC_IN void ForAll(const Vec3u& min, const Vec3u& max, const CLB& clb) 
	{
		auto e = e_SpatialLinkedMap_volume_iterator<T>(*this, min, max, true);
		for (auto it = e_SpatialLinkedMap_volume_iterator<T>(*this, min, max, false); it != e; ++it)
			clb(it.getDataIdx(), *it);
	}

	template<typename CLB> CUDA_FUNC_IN void ForAllCells(const Vec3u& min, const Vec3u& max, const CLB& clb)
	{
		Vec3u min_cell = hashMap.Transform(min_disk), max_cell = hashMap.Transform(max_disk);
		for (unsigned int ax = min_cell.x; ax <= max_cell.x; ax++)
			for (unsigned int ay = min_cell.y; ay <= max_cell.y; ay++)
				for (unsigned int az = min_cell.z; az <= max_cell.z; az++)
				{
					clb(Vec3u(ax,ay,z));
				}
	}

	//internal

	CUDA_FUNC_IN unsigned int idx(const Vec3u& i) const
	{
		return deviceMap[hashMap.Hash(i)];
	}

	CUDA_FUNC_IN unsigned int nextIdx(unsigned int idx) const
	{
		return deviceData[idx].nextIdx;
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return deviceData[idx].value;
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return deviceData[idx].value;
	}
};

template<typename T> struct e_SpatialLinkedMap_volume_iterator
{
	e_SpatialLinkedMap<T>& map;
	Vec3u low, high, diff;// := [low, high)
	unsigned int dataIdx, flatGridIdx;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator(const e_SpatialLinkedMap<T>& m, const Vec3u& mi, const Vec3u& ma, bool isEnd)
		: map((e_SpatialLinkedMap<T>&)m), low(mi), high(ma + Vec3u(1)), diff(high - low)
	{
		flatGridIdx = isEnd ? diff.x * diff.y * diff.z : 0;
		dataIdx = isEnd ? unsigned int(-1) : m.idx(mi);
		if(!isEnd && dataIdx == unsigned int(-1))
			operator++();
	}
	
	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T>& operator++()
	{
		if(dataIdx != unsigned int(-1))
			dataIdx = map.nextIdx(dataIdx);
		while(dataIdx == unsigned int(-1) && ++flatGridIdx != diff.x * diff.y * diff.z)
		{
			unsigned int slice = diff.x * diff.y, inSlice = flatGridIdx % slice;
			Vec3u mi = low + Vec3u(inSlice % diff.x, inSlice / diff.x, flatGridIdx / slice);
			dataIdx = map.idx(mi);
		}
		return *this;
	}

	CUDA_FUNC_IN const T& operator*() const
	{
		return map(dataIdx);
	}

	CUDA_FUNC_IN const T* operator->() const
	{
		return &map(dataIdx);
	}

	CUDA_FUNC_IN T& operator*()
	{
		return map(dataIdx);
	}

	CUDA_FUNC_IN T* operator->()
	{
		return &map(dataIdx);
	}

	CUDA_FUNC_IN bool operator==(const e_SpatialLinkedMap_volume_iterator<T>& rhs) const
	{
		return dataIdx == rhs.dataIdx && flatGridIdx == rhs.flatGridIdx;
	}

	CUDA_FUNC_IN bool operator!=(const e_SpatialLinkedMap_volume_iterator<T>& rhs) const
	{
		return !operator==(rhs);
	}

	CUDA_FUNC_IN unsigned int getDataIdx() const
	{
		return dataIdx;
	}
};

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::begin(const Vec3f& p) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(p), hashMap.Transform(p), false);
}

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::end(const Vec3f& p) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(p), hashMap.Transform(p), true);
}

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::begin(const Vec3f& min, const Vec3f& max) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(min), hashMap.Transform(max), false);
}

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::end(const Vec3f& min, const Vec3f& max) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(min), hashMap.Transform(max), true);
}

//a mapping from R^3 -> T, ie. associating one element with each point in the grid
template<typename T> struct e_SpatialSet
{
	unsigned int gridSize;
	T* deviceData;
	k_HashGrid_Reg hashMap;
public:
	e_SpatialSet(){}
	e_SpatialSet(unsigned int gridSize)
		: gridSize(gridSize)
	{
		CUDA_MALLOC(&deviceData, sizeof(T) * gridSize * gridSize * gridSize);
	}
	
	void Free()
	{
		CUDA_FREE(deviceData);
	}

	void SetSceneDimensions(const AABB& box, float initialRadius)
	{
		hashMap = k_HashGrid_Reg(box, initialRadius, gridSize * gridSize * gridSize);
	}

	void ResetBuffer()
	{
		ThrowCudaErrors(cudaMemset(deviceData, 0, sizeof(T) * gridSize * gridSize * gridSize));
	}

	CUDA_FUNC_IN const T& operator()(const Vec3f& p) const
	{
		return deviceData[hashMap.Hash(p)].value;
	}

	CUDA_FUNC_IN T& operator()(const Vec3f& p)
	{
		return deviceData[hashMap.Hash(p)];
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return deviceData[idx];
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return deviceData[idx];
	}

	CUDA_FUNC_IN unsigned int NumEntries()
	{
		return gridSize * gridSize * gridSize;
	}
};
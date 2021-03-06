#pragma once

#include "IBlockSampler_device.h"
#include "IBlockSampler.h"
#include <Engine/Image.h>
#include <algorithm>

namespace CudaTracerLib
{

class UniformBlockSampler : public IUserPreferenceSampler
{
	std::vector<int> m_indices;
	float m_perBlocksToDraw;
	bool m_nonZero;
public:
	UniformBlockSampler(unsigned int w, unsigned int h)
		: IUserPreferenceSampler(w, h), m_perBlocksToDraw(1), m_nonZero(false)
	{
		int n(0);
		m_indices.resize(getNumTotalBlocks());
		std::generate(std::begin(m_indices), std::end(m_indices), [&] { return n++; });
	}

	virtual void Free()
	{
	}

	virtual IBlockSampler* CreateForSize(unsigned int w, unsigned int h)
	{
		return new UniformBlockSampler(w, h);
	}

	float getPerBlocksToDraw() const
	{
		return m_perBlocksToDraw;
	}

	void setPerBlocksToDraw(float f)
	{
		m_perBlocksToDraw = f;
	}

	virtual void AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer)
	{
		IUserPreferenceSampler::AddPass(img, tracer, varBuffer);

		std::sort(std::begin(m_indices), std::end(m_indices), [&](int i1, int i2) { m_nonZero |= m_userWeights[i1] != 1 || m_userWeights[i2] != 1; return m_userWeights[i1] > m_userWeights[i2]; });
	}

	virtual void IterateBlocks(iterate_blocks_clb_t clb) const
	{
		if (m_nonZero)
		{
			unsigned int idx = 0, N = (unsigned int)(m_indices.size() * min(m_perBlocksToDraw, 1.0f));
			while (idx < N)
			{
				auto flattened_idx = m_indices[idx];

				//do not draw any blocks which the user deselects
				if (m_userWeights[flattened_idx] <= 0)
					break;

				int block_x, block_y, x, y, bw, bh;
				getIdxComponents(flattened_idx, block_x, block_y);

				getBlockRect(block_x, block_y, x, y, bw, bh);

				clb(flattened_idx, x, y, bw, bh);

				idx++;
			}
		}
		else IterateAllBlocksUniform(clb);
	}
};

}
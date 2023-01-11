#pragma once

#include "vectorimage.h"
#include <iplib/common.h>
#include "padding.h"
#include <iplib/parallel.h>

namespace ip
{
	enum class BlockSplitPaddingMode
	{
		Duplicate
	};

#ifdef LEGACY
	enum class BlockSplitLayout
	{
		NotSet,
		Layout4x1,
		Layout2x2,
		Layout1x4,
	};
#else
	enum class BlockSplitLayout
	{
		NotSet,
		Layout8x1,
		Layout4x2,
		Layout2x4,
		Layout1x8
	};
#endif

	struct BlockSplitConfiguration
	{
		int paddingLeft, paddingTop, paddingRight, paddingBottom;

		BlockSplitConfiguration(int paddingLeft, int paddingTop, int paddingRight, int paddingBottom)
			: paddingLeft(paddingLeft), paddingTop(paddingTop), paddingRight(paddingRight), paddingBottom(paddingBottom) {}

		BlockSplitConfiguration(int paddingHorz, int paddingVert)
			: BlockSplitConfiguration(paddingHorz, paddingVert, paddingHorz, paddingVert) {}

		BlockSplitConfiguration(int padding)
			: BlockSplitConfiguration(padding, padding) {}

		BlockSplitConfiguration()
			: BlockSplitConfiguration(0) {}
	};

	// ==================================================================================================

	class BlockSplit
	{
	public:
		inline void SetInputPadding(BlockSplitConfiguration padding, BlockSplitPaddingMode paddingMode);
		inline void SetOutputPadding(BlockSplitConfiguration padding);
		inline void SetSplitLayout(BlockSplitLayout layout);
		inline void ComputeSplitLayout(int Width, int Height);

		inline BlockSplitLayout GetSplitLayout() const;

		inline VectorImageFloat Split(const ImageFloat &src, int ManualBlockWidth = 0, int ManualBlockHeight = 0) const;
		inline VectorImageFloatColor Split(const ImageFloatColor &src, int ManualBlockWidth = 0, int ManualBlockHeight = 0) const;
		inline VectorImageFloatColor Split(const ImageByteColor &src, int ManualBlockWidth = 0, int ManualBlockHeight = 0) const;

		inline ImageFloatColor Gather(const VectorImageFloatColor &src) const;
		inline ImageByteColor Gather(const Image<VectorByte> &src) const;

	private:
		BlockSplitConfiguration inputPadding, outputPadding;
		BlockSplitPaddingMode paddingMode = BlockSplitPaddingMode::Duplicate;
		BlockSplitLayout layout = BlockSplitLayout::NotSet;

		template <int HorzBlocks, int VertBlocks>
		inline VectorImageFloatColor PerformSplit(const ImageFloatColor &src, int ManualBlockWidth = 0, int ManualBlockHeight = 0) const;

		template <int HorzBlocks, int VertBlocks>
		inline VectorImageFloatColor PerformSplit(const ImageByteColor &src, int ManualBlockWidth = 0, int ManualBlockHeight = 0) const;

		template <int HorzBlocks, int VertBlocks>
		inline ImageFloatColor PerformGather(const VectorImageFloatColor &src) const;

		template <int HorzBlocks, int VertBlocks>
		inline ImageByteColor PerformGather(const Image<VectorByte> &src) const;

		template <typename PixelType, class SourceImageType, class DestinationImageType>
		inline void Expand(const ImageReadable<PixelType, SourceImageType> &src, ImageWritable<PixelType, DestinationImageType> &dst) const;

		template <typename ScalarPixelType, typename VectorPixelType>
		struct Helper;

		template <typename PixelType, int HorzBlocks, int VertBlocks, int BlockIndex>
		class SplitReader;
	};

	// ==================================================================================================

#pragma region Helper

	template <typename ScalarPixelType, typename VectorPixelType>
	struct BlockSplit::Helper
	{
	public:

#ifdef LEGACY
		static void ToVector(ScalarPixelType v0, ScalarPixelType v1, ScalarPixelType v2, ScalarPixelType v3, VectorPixelType &dst)
		{
			die("Invalid call");
		}

		static void FromVector(const VectorPixelType &src, ScalarPixelType &v0, ScalarPixelType &v1, ScalarPixelType &v2, ScalarPixelType &v3)
		{
			die("Invalid call");
		}
#else
		static void ToVector(ScalarPixelType v0, ScalarPixelType v1, ScalarPixelType v2, ScalarPixelType v3, ScalarPixelType v4, ScalarPixelType v5, ScalarPixelType v6, ScalarPixelType v7, VectorPixelType &dst)
		{
			die("Invalid call");
		}

		static void FromVector(const VectorPixelType &src, ScalarPixelType &v0, ScalarPixelType &v1, ScalarPixelType &v2, ScalarPixelType &v3, ScalarPixelType &v4, ScalarPixelType &v5, ScalarPixelType &v6, ScalarPixelType &v7)
		{
			die("Invalid call");	
		}
#endif
	};

	template <>
	struct BlockSplit::Helper<float, VectorFloat>
	{
	public:
#ifdef LEGACY
		static void ToVector(float v0, float v1, float v2, float v3, VectorFloat &dst)
		{
			dst = float4(v0, v1, v2, v3);
		}

		static void FromVector(const VectorFloat &src, float &v0, float &v1, float &v2, float &v3)
		{
			v0 = src.get_value().m128_f32[0];
			v1 = src.get_value().m128_f32[1];
			v2 = src.get_value().m128_f32[2];
			v3 = src.get_value().m128_f32[3];
		}
#else
		static void ToVector(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, VectorFloat &dst)
		{
			dst = float8(v0, v1, v2, v3, v4, v5, v6, v7);
		}

		static void FromVector(const VectorFloat &src, float &v0, float &v1, float &v2, float &v3, float &v4, float &v5, float &v6, float &v7)
		{
			v0 = src.get_value().m256_f32[0];
			v1 = src.get_value().m256_f32[1];
			v2 = src.get_value().m256_f32[2];
			v3 = src.get_value().m256_f32[3];
			v4 = src.get_value().m256_f32[4];
			v5 = src.get_value().m256_f32[5];
			v6 = src.get_value().m256_f32[6];
			v7 = src.get_value().m256_f32[7];
		}
#endif
	};

#pragma endregion

	// ==================================================================================================

	void BlockSplit::SetInputPadding(BlockSplitConfiguration padding, BlockSplitPaddingMode paddingMode)
	{
		this->inputPadding = padding;
		this->paddingMode = paddingMode;
	}

	void BlockSplit::SetOutputPadding(BlockSplitConfiguration padding)
	{
		this->outputPadding = padding;
	}

	BlockSplitLayout BlockSplit::GetSplitLayout() const
	{
		return this->layout;
	}

	void BlockSplit::SetSplitLayout(BlockSplitLayout layout)
	{
		this->layout = layout;
	}

	void BlockSplit::ComputeSplitLayout(int Width, int Height)
	{
#ifdef LEGACY
		int p1 = Width * 4 + Height;
		int p2 = Width * 2 + Height * 2;
		int p4 = Width + Height * 4;

		if (p1 < p2)
		{
			layout = BlockSplitLayout::Layout4x1;
		}
		else if (p2 < p4)
		{
			layout = BlockSplitLayout::Layout2x2;
		}
		else
		{
			layout = BlockSplitLayout::Layout1x4;
		}
#else
		int p1 = Width * 8 + Height;
		int p2 = Width * 4 + Height * 2;
		int p4 = Width * 2 + Height * 4;
		int p8 = Width + Height * 8;

		if (p1 < p2)
		{
			layout = BlockSplitLayout::Layout8x1;
		}
		else if (p2 < p4)
		{
			layout = BlockSplitLayout::Layout4x2;
		}
		else if (p4 < p8)
		{
			layout = BlockSplitLayout::Layout2x4;
		}
		else
		{
			layout = BlockSplitLayout::Layout1x8;
		}
#endif
	}


	VectorImageFloat BlockSplit::Split(const ImageFloat &src, int ManualBlockWidth, int ManualBlockHeight) const
	{
		die("Not implemented");
	}

	VectorImageFloatColor BlockSplit::Split(const ImageFloatColor &src, int ManualBlockWidth, int ManualBlockHeight) const
	{
		switch (layout)
		{
#ifdef LEGACY
		case BlockSplitLayout::Layout1x4:
			return PerformSplit<1, 4>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout2x2:
			return PerformSplit<2, 2>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout4x1:
			return PerformSplit<4, 1>(src, ManualBlockWidth, ManualBlockHeight);
#else
		case BlockSplitLayout::Layout1x8:
			return PerformSplit<1, 8>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout2x4:
			return PerformSplit<2, 4>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout4x2:
			return PerformSplit<4, 2>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout8x1:
			return PerformSplit<8, 1>(src, ManualBlockWidth, ManualBlockHeight);
#endif

		default:
			die("Invalid BlockSplitLayout value");
		}
	}

	VectorImageFloatColor BlockSplit::Split(const ImageByteColor &src, int ManualBlockWidth, int ManualBlockHeight) const
	{
#ifdef LEGACY
		switch (layout)
		{
		case BlockSplitLayout::Layout1x4:
			return PerformSplit<1, 4>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout2x2:
			return PerformSplit<2, 2>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout4x1:
			return PerformSplit<4, 1>(src, ManualBlockWidth, ManualBlockHeight);
#else
		switch (layout)
		{
		case BlockSplitLayout::Layout1x8:
			return PerformSplit<1, 8>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout2x4:
			return PerformSplit<2, 4>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout4x2:
			return PerformSplit<4, 2>(src, ManualBlockWidth, ManualBlockHeight);

		case BlockSplitLayout::Layout8x1:
			return PerformSplit<8, 1>(src, ManualBlockWidth, ManualBlockHeight);
#endif
		default:
			die("Invalid BlockSplitLayout value");
		}
	}

	template <typename PixelType, class SourceImageType, class DestinationImageType>
	void BlockSplit::Expand(const ImageReadable<PixelType, SourceImageType> &src, ImageWritable<PixelType, DestinationImageType> &dst) const
	{
		for (int j = 0; j < inputPadding.paddingTop; j++)
		{
			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Left, padding::PaddingAccessMode::Left> p00(src);
			for (int i = 0; i < inputPadding.paddingLeft; i++)
				dst(i, j) = p00(i - inputPadding.paddingLeft, j - inputPadding.paddingTop);

			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Normal, padding::PaddingAccessMode::Left> p10(src);
			for (int i = 0; i < src.Width(); i++)
				dst(i + inputPadding.paddingLeft, j) = p10(i, j - inputPadding.paddingTop);

			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Right, padding::PaddingAccessMode::Left> p20(src);
			for (int i = src.Width() + inputPadding.paddingLeft; i < dst.Width(); i++)
				dst(i, j) = p20(i - inputPadding.paddingLeft, j - inputPadding.paddingTop);
		}

		Parallel::For(0, src.Height(), [this, &src, &dst](int j)
		{
			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Left, padding::PaddingAccessMode::Normal> p01(src);
			for (int i = 0; i < inputPadding.paddingLeft; i++)
				dst(i, j + inputPadding.paddingTop) = p01(i - inputPadding.paddingLeft, j);

			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Normal, padding::PaddingAccessMode::Normal> p11(src);
			for (int i = 0; i < src.Width(); i++)
				dst(i + inputPadding.paddingLeft, j + inputPadding.paddingTop) = p11(i, j);

			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Right, padding::PaddingAccessMode::Normal> p21(src);
			for (int i = src.Width() + inputPadding.paddingLeft; i < dst.Width(); i++)
				dst(i, j + inputPadding.paddingTop) = p21(i - inputPadding.paddingLeft, j);
		});

		for (int j = src.Height() + inputPadding.paddingTop; j < dst.Height(); j++)
		{
			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Left, padding::PaddingAccessMode::Right> p02(src);
			for (int i = 0; i < inputPadding.paddingLeft; i++)
				dst(i, j) = p02(i - inputPadding.paddingLeft, j - inputPadding.paddingTop);

			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Normal, padding::PaddingAccessMode::Right> p12(src);
			for (int i = 0; i < src.Width(); i++)
				dst(i + inputPadding.paddingLeft, j) = p12(i, j - inputPadding.paddingTop);

			padding::DuplicatePaddingAccessor<PixelType, SourceImageType, padding::PaddingAccessMode::Right, padding::PaddingAccessMode::Right> p22(src);
			for (int i = src.Width() + inputPadding.paddingLeft; i < dst.Width(); i++)
				dst(i, j) = p22(i - inputPadding.paddingLeft, j - inputPadding.paddingTop);
		}
	}

	template <int HorzBlocks, int VertBlocks>
	VectorImageFloatColor BlockSplit::PerformSplit(const ImageFloatColor &src, int ManualBlockWidth, int ManualBlockHeight) const
	{
#ifdef LEGACY
		static_assert(HorzBlocks * VertBlocks == 4, "The number of blocks should be equal to 4");
#else
		static_assert(HorzBlocks * VertBlocks == 8, "The number of blocks should be equal to 8");
#endif
		int BlockWidth = ManualBlockWidth == 0 ? (src.Width() + HorzBlocks - 1) / HorzBlocks : ManualBlockWidth;
		int BlockHeight = ManualBlockHeight == 0 ? (src.Height() + VertBlocks - 1) / VertBlocks : ManualBlockHeight;

		ImageFloatColor tmp(BlockWidth * HorzBlocks + inputPadding.paddingLeft + inputPadding.paddingRight, BlockHeight * VertBlocks + inputPadding.paddingTop + inputPadding.paddingBottom);

		Expand(src, tmp);

		VectorImageFloatColor res(BlockWidth + inputPadding.paddingLeft + inputPadding.paddingRight, BlockHeight + inputPadding.paddingTop + inputPadding.paddingBottom);

		Parallel::For(0, res.Height(), [BlockWidth, BlockHeight, &tmp, &res](int j)
		{
			for (int i = 0; i < res.Width(); i++)
			{
#ifdef LEGACY
				Helper<PixelFloatRGBA, VectorFloatColor>::ToVector(
					tmp(i, j),
					tmp(i + BlockWidth * (1 % HorzBlocks), j + BlockHeight * (1 / HorzBlocks)),
					tmp(i + BlockWidth * (2 % HorzBlocks), j + BlockHeight * (2 / HorzBlocks)),
					tmp(i + BlockWidth * (3 % HorzBlocks), j + BlockHeight * (3 / HorzBlocks)),
					res(i, j));
#else
				Helper<PixelFloatRGBA, VectorFloatColor>::ToVector(
					tmp(i, j),
					tmp(i + BlockWidth * (1 % HorzBlocks), j + BlockHeight * (1 / HorzBlocks)),
					tmp(i + BlockWidth * (2 % HorzBlocks), j + BlockHeight * (2 / HorzBlocks)),
					tmp(i + BlockWidth * (3 % HorzBlocks), j + BlockHeight * (3 / HorzBlocks)),
					tmp(i + BlockWidth * (4 % HorzBlocks), j + BlockHeight * (4 / HorzBlocks)),
					tmp(i + BlockWidth * (5 % HorzBlocks), j + BlockHeight * (5 / HorzBlocks)),
					tmp(i + BlockWidth * (6 % HorzBlocks), j + BlockHeight * (6 / HorzBlocks)),
					tmp(i + BlockWidth * (7 % HorzBlocks), j + BlockHeight * (7 / HorzBlocks)),
					res(i, j));
#endif
			}
		});

		return res;
	}

	template <int HorzBlocks, int VertBlocks>
	VectorImageFloatColor BlockSplit::PerformSplit(const ImageByteColor &src, int ManualBlockWidth, int ManualBlockHeight) const
	{
#ifdef LEGACY
		static_assert(HorzBlocks * VertBlocks == 4, "The number of blocks should be equal to 4");
#else
		static_assert(HorzBlocks * VertBlocks == 8, "The number of blocks should be equal to 8");
#endif

		static const VectorFloat YR(0.299f);
		static const VectorFloat YG(0.587f);
		static const VectorFloat YB(0.114f);

		static const VectorFloat VR(-0.169f);
		static const VectorFloat VG(-0.331f);
		static const VectorFloat VB(0.499f);

		static const VectorFloat UR(0.499f);
		static const VectorFloat UG(-0.418f);
		static const VectorFloat UB(-0.0813f);

		int BlockWidth = ManualBlockWidth == 0 ? (src.Width() + HorzBlocks - 1) / HorzBlocks : ManualBlockWidth;
		int BlockHeight = ManualBlockHeight == 0 ? (src.Height() + VertBlocks - 1) / VertBlocks : ManualBlockHeight;

		ImageByteColor tmp(BlockWidth * HorzBlocks + inputPadding.paddingLeft + inputPadding.paddingRight, BlockHeight * VertBlocks + inputPadding.paddingTop + inputPadding.paddingBottom);

		Expand(src, tmp);

		VectorImageFloatColor res(BlockWidth + inputPadding.paddingLeft + inputPadding.paddingRight, BlockHeight + inputPadding.paddingTop + inputPadding.paddingBottom);

		Parallel::For(0, res.Height(), [BlockWidth, BlockHeight, &tmp, &res](int j)
		{
			for (int i = 0; i < res.Width(); i++)
			{
				float4 v0 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (0 % HorzBlocks), j + BlockHeight * (0 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();

				float4 v1 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (1 % HorzBlocks), j + BlockHeight * (1 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();

				float4 v2 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (2 % HorzBlocks), j + BlockHeight * (2 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();

				float4 v3 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (3 % HorzBlocks), j + BlockHeight * (3 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();

#ifndef LEGACY
				float4 v4 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (4 % HorzBlocks), j + BlockHeight * (4 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();

				float4 v5 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (5 % HorzBlocks), j + BlockHeight * (5 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();

				float4 v6 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (6 % HorzBlocks), j + BlockHeight * (6 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();

				float4 v7 = float4::load_single(tmp.pixeladdr(i + BlockWidth * (7 % HorzBlocks), j + BlockHeight * (7 / HorzBlocks)))
					.cast_byte16().convert_u32().convert_f32();
#endif

				float4 bg02 = v0.unpack_lo(v2);
				float4 ra02 = v0.unpack_hi(v2);
				float4 bg13 = v1.unpack_lo(v3);
				float4 ra13 = v1.unpack_hi(v3);

				float4 b0123 = bg02.unpack_lo(bg13);
				float4 g0123 = bg02.unpack_hi(bg13);
				float4 r0123 = ra02.unpack_lo(ra13);

#ifndef LEGACY
				float4 bg46 = v4.unpack_lo(v6);
				float4 ra46 = v4.unpack_hi(v6);
				float4 bg57 = v5.unpack_lo(v7);
				float4 ra57 = v5.unpack_hi(v7);

				float4 b4567 = bg46.unpack_lo(bg57);
				float4 g4567 = bg46.unpack_hi(bg57);
				float4 r4567 = ra46.unpack_lo(ra57);

				float8 b = float8(b0123, b4567);
				float8 g = float8(g0123, g4567);
				float8 r = float8(r0123, r4567);

				res(i, j) = VectorFloatColor(r * YR + g * YG + b * YB, r * UR + g * UG + b * UB, r * VR + g * VG + b * VB);
#else
				res(i, j) = VectorFloatColor(r0123 * YR + g0123 * YG + b0123 * YB, r0123 * UR + g0123 * UG + b0123 * UB, r0123 * VR + g0123 * VG + b0123 * VB);
#endif
			}
		});

		return res;
	}

	ImageFloatColor BlockSplit::Gather(const VectorImageFloatColor &src) const
	{
		switch (layout)
		{
#ifdef LEGACY
		case BlockSplitLayout::Layout1x4:
			return PerformGather<1, 4>(src);

		case BlockSplitLayout::Layout2x2:
			return PerformGather<2, 2>(src);

		case BlockSplitLayout::Layout4x1:
			return PerformGather<4, 1>(src);
#else
		case BlockSplitLayout::Layout1x8:
			return PerformGather<1, 8>(src);

		case BlockSplitLayout::Layout2x4:
			return PerformGather<2, 4>(src);

		case BlockSplitLayout::Layout4x2:
			return PerformGather<4, 2>(src);

		case BlockSplitLayout::Layout8x1:
			return PerformGather<8, 1>(src);
#endif
		default:
			die("Invalid BlockSplitLayout value");
		}
	}

	ImageByteColor BlockSplit::Gather(const Image<VectorByte> &src) const
	{
		switch (layout)
		{
#ifdef LEGACY
		case BlockSplitLayout::Layout1x4:
			return PerformGather<1, 4>(src);

		case BlockSplitLayout::Layout2x2:
			return PerformGather<2, 2>(src);

		case BlockSplitLayout::Layout4x1:
			return PerformGather<4, 1>(src);
#else
		case BlockSplitLayout::Layout1x8:
			return PerformGather<1, 8>(src);

		case BlockSplitLayout::Layout2x4:
			return PerformGather<2, 4>(src);

		case BlockSplitLayout::Layout4x2:
			return PerformGather<4, 2>(src);

		case BlockSplitLayout::Layout8x1:
			return PerformGather<8, 1>(src);
#endif
		default:
			die("Invalid BlockSplitLayout value");
		}
	}

	template <int HorzBlocks, int VertBlocks>
	ImageFloatColor BlockSplit::PerformGather(const VectorImageFloatColor &src) const
	{
#ifdef LEGACY
		static_assert(HorzBlocks * VertBlocks == 4, "The number of blocks should be equal to 4");
#else
		static_assert(HorzBlocks * VertBlocks == 8, "The number of blocks should be equal to 8");
#endif
		int BlockWidth = src.Width() - outputPadding.paddingLeft - outputPadding.paddingRight;
		int BlockHeight = src.Height() - outputPadding.paddingTop - outputPadding.paddingBottom;

		ImageFloatColor res(BlockWidth * HorzBlocks, BlockHeight * VertBlocks);

		Parallel::For(0, BlockHeight, [this, &src, &res, BlockWidth, BlockHeight](int j)
		{
			for (int i = 0; i < BlockWidth; i += 2)
			{
				VectorFloat b0_7a = src(i + outputPadding.paddingLeft, j + outputPadding.paddingTop).y;
				VectorFloat g0_7a = src(i + outputPadding.paddingLeft, j + outputPadding.paddingTop).u;
				VectorFloat r0_7a = src(i + outputPadding.paddingLeft, j + outputPadding.paddingTop).v;
				VectorFloat zero = VectorFloat::zero();

				VectorFloat bg01_45a = b0_7a.unpack_lo(g0_7a);
				VectorFloat ra01_45a = r0_7a.unpack_lo(zero);
				VectorFloat bg23_67a = b0_7a.unpack_hi(g0_7a);
				VectorFloat ra23_67a = r0_7a.unpack_hi(zero);

				VectorFloat bgra0_4a = bg01_45a.shuffle<0x44>(ra01_45a);
				VectorFloat bgra1_5a = bg01_45a.shuffle<0xEE>(ra01_45a);
				VectorFloat bgra2_6a = bg23_67a.shuffle<0x44>(ra23_67a);
				VectorFloat bgra3_7a = bg23_67a.shuffle<0xEE>(ra23_67a);

				VectorFloat b0_7b = src(i + outputPadding.paddingLeft + 1, j + outputPadding.paddingTop).y;
				VectorFloat g0_7b = src(i + outputPadding.paddingLeft + 1, j + outputPadding.paddingTop).u;
				VectorFloat r0_7b = src(i + outputPadding.paddingLeft + 1, j + outputPadding.paddingTop).v;

				VectorFloat bg01_45b = b0_7b.unpack_lo(g0_7b);
				VectorFloat ra01_45b = r0_7b.unpack_lo(zero);
				VectorFloat bg23_67b = b0_7b.unpack_lo(g0_7b);
				VectorFloat ra23_67b = r0_7b.unpack_lo(zero);

				VectorFloat bgra0_4b = bg01_45b.shuffle<0x44>(ra01_45b);
				VectorFloat bgra1_5b = bg01_45b.shuffle<0xEE>(ra01_45b);
				VectorFloat bgra2_6b = bg23_67b.shuffle<0x44>(ra23_67b);
				VectorFloat bgra3_7b = bg23_67b.shuffle<0xEE>(ra23_67b);

#ifdef LEGACY
				bgra0_4a.stream(res.pixeladdr(i + BlockWidth * (0 % HorzBlocks), j + BlockHeight * (0 / HorzBlocks)));
				bgra1_5a.stream(res.pixeladdr(i + BlockWidth * (1 % HorzBlocks), j + BlockHeight * (1 / HorzBlocks)));
				bgra2_6a.stream(res.pixeladdr(i + BlockWidth * (2 % HorzBlocks), j + BlockHeight * (2 / HorzBlocks)));
				bgra3_7a.stream(res.pixeladdr(i + BlockWidth * (3 % HorzBlocks), j + BlockHeight * (3 / HorzBlocks)));

				bgra0_4b.stream(res.pixeladdr(i + BlockWidth * (0 % HorzBlocks) + 1, j + BlockHeight * (0 / HorzBlocks)));
				bgra1_5b.stream(res.pixeladdr(i + BlockWidth * (1 % HorzBlocks) + 1, j + BlockHeight * (1 / HorzBlocks)));
				bgra2_6b.stream(res.pixeladdr(i + BlockWidth * (2 % HorzBlocks) + 1, j + BlockHeight * (2 / HorzBlocks)));
				bgra3_7b.stream(res.pixeladdr(i + BlockWidth * (3 % HorzBlocks) + 1, j + BlockHeight * (3 / HorzBlocks)));

#else
				bgra0_4a.permute128<0x20>(bgra0_4b).stream(res.pixeladdr(i + BlockWidth * (0 % HorzBlocks), j + BlockHeight * (0 / HorzBlocks)));
				bgra1_5a.permute128<0x20>(bgra1_5b).stream(res.pixeladdr(i + BlockWidth * (1 % HorzBlocks), j + BlockHeight * (1 / HorzBlocks)));
				bgra2_6a.permute128<0x20>(bgra2_6b).stream(res.pixeladdr(i + BlockWidth * (2 % HorzBlocks), j + BlockHeight * (2 / HorzBlocks)));
				bgra3_7a.permute128<0x20>(bgra3_7b).stream(res.pixeladdr(i + BlockWidth * (3 % HorzBlocks), j + BlockHeight * (3 / HorzBlocks)));
				bgra0_4a.permute128<0x31>(bgra0_4b).stream(res.pixeladdr(i + BlockWidth * (4 % HorzBlocks), j + BlockHeight * (4 / HorzBlocks)));
				bgra1_5a.permute128<0x31>(bgra1_5b).stream(res.pixeladdr(i + BlockWidth * (5 % HorzBlocks), j + BlockHeight * (5 / HorzBlocks)));
				bgra2_6a.permute128<0x31>(bgra2_6b).stream(res.pixeladdr(i + BlockWidth * (6 % HorzBlocks), j + BlockHeight * (6 / HorzBlocks)));
				bgra3_7a.permute128<0x31>(bgra3_7b).stream(res.pixeladdr(i + BlockWidth * (7 % HorzBlocks), j + BlockHeight * (7 / HorzBlocks)));
#endif
			}
		});

		return res;
	}

	template <int HorzBlocks, int VertBlocks>
	ImageByteColor BlockSplit::PerformGather(const Image<VectorByte> &src) const
	{
#ifdef LEGACY
		static_assert(HorzBlocks * VertBlocks == 4, "The number of blocks should be equal to 4");
#else
		static_assert(HorzBlocks * VertBlocks == 8, "The number of blocks should be equal to 8");
#endif
		int BlockWidth = src.Width() - outputPadding.paddingLeft - outputPadding.paddingRight;
		int BlockHeight = src.Height() - outputPadding.paddingTop - outputPadding.paddingBottom;

		ImageByteColor res(BlockWidth * HorzBlocks, BlockHeight * VertBlocks);

		Parallel::For(0, BlockHeight, [this, &src, &res, BlockWidth, BlockHeight](int j)
			// for (int j = 0; j < BlockHeight; j++)
		{
			for (int i = 0; i < BlockWidth; i++)
			{
				PixelByteRGBA *c = (PixelByteRGBA*)src.pixeladdr(i + outputPadding.paddingLeft, j + outputPadding.paddingTop);

				res(i + BlockWidth * (0 % HorzBlocks), j + BlockHeight * (0 / HorzBlocks)) = c[0];
				res(i + BlockWidth * (1 % HorzBlocks), j + BlockHeight * (1 / HorzBlocks)) = c[1];
				res(i + BlockWidth * (2 % HorzBlocks), j + BlockHeight * (2 / HorzBlocks)) = c[2];
				res(i + BlockWidth * (3 % HorzBlocks), j + BlockHeight * (3 / HorzBlocks)) = c[3];
#ifndef LEGACY
				res(i + BlockWidth * (4 % HorzBlocks), j + BlockHeight * (4 / HorzBlocks)) = c[4];
				res(i + BlockWidth * (5 % HorzBlocks), j + BlockHeight * (5 / HorzBlocks)) = c[5];
				res(i + BlockWidth * (6 % HorzBlocks), j + BlockHeight * (6 / HorzBlocks)) = c[6];
				res(i + BlockWidth * (7 % HorzBlocks), j + BlockHeight * (7 / HorzBlocks)) = c[7];
#endif
			}
		});

		return res;
	}
}

#pragma once

namespace ip
{








	public class StructureTensorAnalysis
	{
		public ImageFloat lambda1, lambda2;
		public ImageVector2f dir;

		public StructureTensorAnalysis(Image<float> img, float der_sigma = 0.7f, float tensor_sigma = 1.0f, int wsize = 6)
		{
			StructureTensorAnalysisFilters staf = new StructureTensorAnalysisFilters(der_sigma, tensor_sigma, wsize);

			ImageFloat Ix = img.FilterHorizontal(staf.der_filter, wsize / 2).Instance().FilterVertical(staf.n_filter, wsize / 2);
			ImageFloat Iy = img.FilterHorizontal(staf.n_filter, wsize / 2).Instance().FilterVertical(staf.der_filter, wsize / 2);

			ImageFloat A = Ix.Sqr().Instance().FilterHorizontal(staf.t_filter, wsize / 2 - 1).Instance().FilterVertical(staf.t_filter, wsize / 2 - 1);
			ImageFloat B = Iy.Sqr().Instance().FilterHorizontal(staf.t_filter, wsize / 2 - 1).Instance().FilterVertical(staf.t_filter, wsize / 2 - 1);
			ImageFloat C = (Ix * Iy).Instance().FilterHorizontal(staf.t_filter, wsize / 2 - 1).Instance().FilterVertical(staf.t_filter, wsize / 2 - 1);

			/* ImageFloat Ix = img.DerivativeX(der_sigma);
			ImageFloat Iy = img.DerivativeY(der_sigma);

			ImageFloat A = Ix.Sqr().GaussFilter(tensor_sigma);
			ImageFloat B = Iy.Sqr().GaussFilter(tensor_sigma);
			ImageFloat C = (Ix * Iy).GaussFilter(tensor_sigma); */

			lambda1 = new ImageFloat(img.Width, img.Height);
			lambda2 = new ImageFloat(img.Width, img.Height);
			dir = new ImageVector2f(img.Width, img.Height);

			ParallelExecute.For(0, img.Height, j = >
			{
				for (int i = 0; i < img.Width; i++)
				{
					float l1, l2;
					Vector2f v;

					Analyze(A[i, j], C[i, j], B[i, j],
						out l1, out l2, out v);

					lambda1[i, j] = l1;
					lambda2[i, j] = l2;
					dir[i, j] = v;
				}
			});

			Ix.Dispose();
			Iy.Dispose();
		}

		public static void Analyze(float lxx, float lxy, float lyy, out float l1, out float l2, out Vector2f dir)
		{
			float d = (float)Math.Sqrt((lxx - lyy) * (lxx - lyy) + 4 * lxy * lxy);
			l1 = (lxx + lyy + d) * 0.5f;
			l2 = (lxx + lyy - d) * 0.5f;

			float dx, dy;

			float lmax = Math.Abs(l1) < Math.Abs(l2) ? l1 : l2;

			if (Math.Abs(lxx - lmax) > Math.Abs(lyy - lmax))
			{
				dx = lxy;
				dy = lmax - lxx;
			}
			else
			{
				dx = lmax - lyy;
				dy = lxy;
			}

			d = (float)Math.Sqrt(dx * dx + dy * dy);

			if (d > 1e-6f)
				dir = new Vector2f(dx / d, dy / d);
			else
				dir = new Vector2f(0.0f, 0.0f);
		}
	}














}
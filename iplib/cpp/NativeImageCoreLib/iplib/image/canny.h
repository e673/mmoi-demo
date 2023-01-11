#pragma once

#include <iplib/image/core.h>
#include <iplib/image/filter.h>

namespace ip
{
	class Canny
		: public ImageReadable<float, Canny>
	{
	public:
		template <typename PixelType, class ImageType>
		Canny(const ImageReadable<PixelType, ImageType> &img, float sigma);

		int Width() const;
		int Height() const;
		float pixel(int x, int y) const;

		Image<PixelFloatVector> GetGradient() const;

	private:
		Image<float> dx, dy, grad, nms, res;

		void NonMaximumSuppression();
	};

	// ==================================================================================================

	template <typename PixelType, class ImageType>
	Canny::Canny(const ImageReadable<PixelType, ImageType> &img, float sigma)
		: dx(img.Width(), img.Height())
		, dy(img.Width(), img.Height())
		, grad(img.Width(), img.Height())
		, nms(img.Width(), img.Height())
		, res(img.Width(), img.Height())
	{
		DerivativeX(img, dx, sigma);
		DerivativeY(img, dy, sigma);

		for (int j = 0; j < img.Height(); j++)
			for (int i = 0; i < img.Width(); i++)
			{
				grad(i, j) = sqrtf(dx(i, j) * dx(i, j) + dy(i, j) * dy(i, j));
				nms(i, j) = 0.0f;
			}

		NonMaximumSuppression();

		for (int j = 0; j < img.Height(); j++)
			for (int i = 0; i < img.Width(); i++)
				res(i, j) = nms(i, j);
	}

	int Canny::Width() const
	{
		return res.Width();
	}

	int Canny::Height() const
	{
		return res.Height();
	}

	float Canny::pixel(int x, int y) const
	{
		return res.pixel(x, y);
	}

	Image<PixelFloatVector> Canny::GetGradient() const
	{
		Image<PixelFloatVector> res(res.Width(), res.Height());

		for (int j = 0; j < res.Height(); j++)
			for (int i = 0; i < res.Width(); i++)
				res(i, j) = PixelFloatVector(dx(i, j), dy(i, j));

		return res;
	}

	void Canny::NonMaximumSuppression()
	{
		for (int j = 1; j < res.Height() - 1; j++)
		{
			for (int i = 1; i < res.Width() - 1; i++)
			{
				float gval = grad(i, j);
				float vx = dx(i, j);
				float vy = dy(i, j);

				float tan = (vx == 0.0f ? 10000.0f : vy / vx);

				if (tan > 2.5f || tan < -2.5f)
				{
					if (gval > grad(i, j - 1) && gval > grad(i, j + 1))
						nms(i, j) = gval;
				}
				else if (tan > 0.4f && tan <= 2.5f)
				{
					if (gval > grad(i - 1, j - 1) && gval > grad(i + 1, j + 1))
						nms(i, j) = gval;
				}
				else if (tan > -0.4f && tan <= 0.4f)
				{
					if (gval > grad(i - 1, j) && gval > grad(i + 1, j))
						nms(i, j) = gval;
				}
				else
				{
					if (gval > grad(i - 1, j + 1) && gval > grad(i + 1, j - 1))
						nms(i, j) = gval;
				}
			}
		}
	}


	/*



	float gval = grad[x, y];



	public override void GetOperands(List<ImageBase> list)
	{
	list.Add(this);
	dx.GetOperands(list);
	dy.GetOperands(list);
	grad.GetOperands(list);
	}

	protected override RuntimeType CreateRuntimeTypeAndGetParameters(List<object> param, bool write)
	{
	param.Add(this);
	return CreateRuntimeTypeFunction<NonMaximumSuppression>(dx, dy, grad);
	}

	protected override void CreateExpressionData(RuntimeExpressionData data, bool write)
	{
	ParameterExpression this_img = data.CreateParameter<NonMaximumSuppression>();
	ParameterExpression w = data.CreateVariable<int>(Expression.Decrement(Expression.Call(this_img, typeof(NonMaximumSuppression).GetMethod("get_Width"))));
	ParameterExpression h = data.CreateVariable<int>(Expression.Decrement(Expression.Call(this_img, typeof(NonMaximumSuppression).GetMethod("get_Height"))));

	ParameterExpression gval = Expression.Parameter(typeof(float));
	ParameterExpression tan = Expression.Parameter(typeof(float));


	data.GetPixel2D = (x, y) => Expression.Block(new ParameterExpression[] { tan, gval },
	Expression.Condition(Expression.Or(Expression.Or(Expression.Or(Expression.LessThan(x, Expression.Constant(1)),
	Expression.LessThan(y, Expression.Constant(1))), Expression.GreaterThanOrEqual(x, w)), Expression.GreaterThanOrEqual(y, h)),
	Expression.Constant(0.0f),
	Expression.Block(
	Expression.Assign(gval, grad.GetPixelExpression(x, y)),
	Expression.Assign(tan, Expression.Condition(Expression.Equal(dx.GetPixelExpression(x, y), Expression.Constant(0.0f)),
	Expression.Constant(10000.0f),
	Expression.Divide(dy.GetPixelExpression(x, y), dx.GetPixelExpression(x, y)))),
	Expression.Condition(Expression.Or(Expression.GreaterThan(tan, Expression.Constant(2.5f)), Expression.LessThan(tan, Expression.Constant(-2.5f))),
	Expression.Condition(Expression.And(Expression.GreaterThan(gval, grad.GetPixelExpression(x, Expression.Decrement(y))),
	Expression.GreaterThan(gval, grad.GetPixelExpression(x, Expression.Increment(y)))),
	gval, Expression.Constant(0.0f)),
	Expression.Condition(Expression.And(Expression.GreaterThan(tan, Expression.Constant(0.4f)), Expression.LessThanOrEqual(tan, Expression.Constant(2.5f))),
	Expression.Condition(Expression.And(Expression.GreaterThan(gval, grad.GetPixelExpression(Expression.Decrement(x), Expression.Decrement(y))),
	Expression.GreaterThan(gval, grad.GetPixelExpression(Expression.Increment(x), Expression.Increment(y)))),
	gval, Expression.Constant(0.0f)),
	Expression.Condition(Expression.And(Expression.GreaterThan(tan, Expression.Constant(-0.4f)), Expression.LessThanOrEqual(tan, Expression.Constant(0.4f))),
	Expression.Condition(Expression.And(Expression.GreaterThan(gval, grad.GetPixelExpression(Expression.Decrement(x), y)),
	Expression.GreaterThan(gval, grad.GetPixelExpression(Expression.Increment(x), y))),
	gval, Expression.Constant(0.0f)),
	Expression.Condition(Expression.And(Expression.GreaterThan(gval, grad.GetPixelExpression(Expression.Decrement(x), Expression.Increment(y))),
	Expression.GreaterThan(gval, grad.GetPixelExpression(Expression.Increment(x), Expression.Decrement(y)))),
	gval, Expression.Constant(0.0f))))))));
	}

	public override string GetTypeName()
	{
	return string.Format("NonMaximumSuppression({0}, {1}, {2})", dx.GetTypeName(), dy.GetTypeName(), grad.GetTypeName());
	}
	}








	public class Canny
	{
		Image<float> img;
		ImageFloat dx, dy, grad, nms, res;

		private Canny() { }

		public static Canny Use(Image<float> img)
		{
			Canny canny = new Canny();
			canny.img = img;
			return canny;
		}

		public Canny DetectAll(float sigma)
		{
			dx = img.DerivativeX(sigma);
			dy = img.DerivativeY(sigma);
			grad = (dx * dx + dy * dy).Sqrt().Cast<float>();
			res = nms = NonMaximumSuppression(dx, dy, grad);
			return this;
		}

		public Canny ApplyAutoThreshold(float thr_tweak = 1.0f)
		{
			res = new ImageFloat(grad.Width, grad.Height);

			float max_g2 = grad.Max();
			float thr1 = max_g2 * 0.2f * thr_tweak;     // The most strongest line segments
			float thr2 = thr1 * 0.1f;                   // Other line segments, maximum canny edge criterion
			float thr3 = thr2 * 0.1f;                   // Minimum canny edge criterion

			Hysteresis(nms, res, thr1, thr2, thr3);

			return this;
		}

		public Canny Detect(float sigma, float thr_tweak = 1.0f)
		{
			return DetectAll(sigma).ApplyAutoThreshold(thr_tweak);
		}

		public FunctionalImage<bool> ResultMask()
		{
			return res.GreaterMask(0.0f);
		}

		public FunctionalImage<float> Result()
		{
			return res.Wrap();
		}

		public ImageFloat GradientX{ get{ return dx; } }

		public ImageFloat GradientY{ get{ return dy; } }

		public ImageFloat GradientModulus{ get{ return grad; } }

			public FunctionalImage<Vector2f> GetGradient()
		{
			return dx.VectorX() + dy.VectorY();
		}

		public ImageFloat AllEdges{ get{ return nms; } }


			public static FunctionalImage<float> NonMaximumSuppression(Image<float> dx, Image<float> dy, Image<float> grad)
		{
			return new NonMaximumSuppression(dx, dy, grad);
		}

		public static void Hysteresis(Image<float> grad, Image<float> res, float thr_strongest, float thr_strong, float thr_weak)
		{
			Size size = res.MatchSize(grad);

			res.Assign(grad.GreaterMask(thr_strongest).ConditionalConvert(255.0f,
				grad.GreaterMask(thr_strong).ConditionalConvert(128.0f,
				grad.GreaterMask(thr_weak).ConditionalConvert(1.0f, 0.0f))));

			Queue<Point> qp = new Queue<Point>();
			for (int j = 0; j < size.Height; j++)
				for (int i = 0; i < size.Width; i++)
					if (grad[i, j] > thr_strong)
						qp.Enqueue(new Point(i, j));

			// Гистерезис
			while (qp.Count > 0)
			{
				Point point = qp.Dequeue();
				if (point.X > 0 && res[point.X - 1, point.Y] == 1.0f && grad[point.X - 1, point.Y] > thr_weak)
				{
					res[point.X - 1, point.Y] = 64.0f;
					qp.Enqueue(new Point(point.X - 1, point.Y));
				}
				if (point.X < grad.Width - 1 && res[point.X + 1, point.Y] == 1.0f && grad[point.X + 1, point.Y] > thr_weak)
				{
					res[point.X + 1, point.Y] = 64.0f;
					qp.Enqueue(new Point(point.X + 1, point.Y));
				}
				if (point.Y > 0 && res[point.X, point.Y - 1] == 1.0f && grad[point.X, point.Y - 1] > thr_weak)
				{
					res[point.X, point.Y - 1] = 64.0f;
					qp.Enqueue(new Point(point.X, point.Y - 1));
				}
				if (point.Y < grad.Height - 1 && res[point.X, point.Y + 1] == 1.0f && grad[point.X, point.Y + 1] > thr_weak)
				{
					res[point.X, point.Y + 1] = 64.0f;
					qp.Enqueue(new Point(point.X, point.Y + 1));
				}
			}
		}
	} */
}
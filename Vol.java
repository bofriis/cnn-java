package rocks.propeller.karpathy;

import com.improaim.shared.math.Matrix32;

/**
 *
 * @author gola
 */
public class Vol {

	public int sx;
	public int sy;
	public int depth;
	public double[] w;
	public double[] dw;

	Util global = new Util();
	
	public double[] predictions() {
		return w;
	}

	public int best() {
		float[] wd = Matrix32.toFloat(w);
		var maxidx =  Matrix32.sumMinMaxSq(wd);
		return maxidx.maxIndexY;
	}

	public Vol(int sx, int sy, int depth, Double c) {
		// we were given dimensions of the vol
		this.sx = sx;
		this.sy = sy;
		this.depth = depth;
		int n = sx * sy * depth;
		this.w = new double[n];
		this.dw = global.zeros(n);
		if (c == null) {
			// weight normalization is done to equalize the output
			// variance of every neuron, otherwise neurons with a lot
			// of incoming connections have outputs of larger variance
			double scale = Math.sqrt(1.0 / (sx * sy * depth));
			for (int i = 0; i < n; i++) {
				this.w[i] = global.randn(0.0, scale);
			}
		} else {
			for (int i = 0; i < n; i++) {
				this.w[i] = c;
			}
		}
	}

	public Vol(double[] sx) {
		// we were given a list in sx, assume 1D volume and fill it up
		this.sx = 1;
		this.sy = 1;
		this.depth = sx.length;
		// we have to do the following copy because we want to use
		// fast typed arrays, not an ordinary javascript array
		this.w = global.zeros(this.depth);
		this.dw = global.zeros(this.depth);
		for (int i = 0; i < this.depth; i++) {
			this.w[i] = sx[i];
		}
	}

	 public double get(int x, int y, int d) {
		int ix = ((this.sx * y) + x) * this.depth + d;
		return this.w[ix];
	}

	public void set(int x, int y, int d, double v) {
		int ix = ((this.sx * y) + x) * this.depth + d;
		this.w[ix] = v;
	}

	void add(int x, int y, int d, double v) {
		int ix = ((this.sx * y) + x) * this.depth + d;
		this.w[ix] += v;
	}

	 public double get_grad(int x, int y, int d) {
		int ix = ((this.sx * y) + x) * this.depth + d;
		return this.dw[ix];
	}

	 public void set_grad(int x, int y, int d, double v) {
		int ix = ((this.sx * y) + x) * this.depth + d;
		this.dw[ix] = v;
	}

	void add_grad(int x, int y, int d, double v) {
		int ix = ((this.sx * y) + x) * this.depth + d;
		this.dw[ix] += v;
	}

	Vol cloneAndZero() {
		return new Vol(this.sx, this.sy, this.depth, 0.0);
	}

	Vol cloneVol() {
		Vol v = new Vol(this.sx, this.sy, this.depth, 0.0);
		int n = this.w.length;
		for (int i = 0; i < n; i++) {
			v.w[i] = this.w[i];
			v.dw[i] = this.dw[i];
		}
		return v;
	}

	void addFrom(Vol v) {
		for (int k = 0; k < this.w.length; k++) {
			this.w[k] += v.w[k];
		}
	}

	void addFromScaled(Vol v, double a) {
		for (int k = 0; k < this.w.length; k++) {
			this.w[k] += a * v.w[k];
		}
	}

	void setConst(double a) {
		for (int k = 0; k < this.w.length; k++) {
			this.w[k] = a;
		}
	}

	public Matrix32 toMatrix(int depthIndex) {
		int w = sx;
		int h = sy;
		Matrix32 m = new Matrix32(w, h, 1);

		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;

		// First pass: find min/max for normalization

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				double v = get(x, y, depthIndex);
				if (v < min)
					min = v;
				if (v > max)
					max = v;
			}
		}

		// Second pass: normalize and copy into matrix
		double range = max - min + 1e-8;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				double v = get(x, y, depthIndex);
				m.set(x, y, (v - min) / range); // normalized 0..1
			}
		}

		return m;
	}

	public Matrix32 toMatrix(int tileSize, int padding) {
		int w = sx;
		int h = sy;

		int cols = (int) Math.ceil(Math.sqrt(depth));
		int rows = (int) Math.ceil((double)depth / cols);
//		int canvasW = cols * (w + padding) - padding;
//		int canvasH = rows * (h + padding) - padding;

		int canvasW = w*depth; // same width as slice
		int canvasH = sy;

		Matrix32 canvas = new Matrix32(canvasH, canvasW );


		// Render each slice in a new horizontal row
		for (int d = 0; d < depth; d++) {
			Matrix32 m = new Matrix32(h, w);
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					double v = get(x, y, d);
					
					m.set( y,x, v);
				}
			}
			canvas.set(0,sx*d, m);
		}

		return canvas;
	}
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import com.improaim.org.json.JSONArray;
import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;
import com.improaim.shared.misc.Multithreading;

/**
 *
 * @author gola
 */
public class ConvLayer implements ILayer {

	int out_depth;
	int sx; // filter size. Should be odd if possible, it's cleaner.
	int in_depth;
	int in_sx;
	int in_sy;
	public Vol[] filters;
	public Vol biases;
	public Vol in_act;
	public Vol out_act;

	int sy;
	int stride; // stride at which we apply filters to input volume
	int pad; // amount of 0 padding to add around borders of input volume
	double l1_decay_mul;
	double l2_decay_mul;

	int out_sx;
	int out_sy;
	String layer_type = "conv";

	Util global = new Util();

	public int get_out_sx() {
		return out_sx;
	}

	public int get_out_sy() {
		return out_sy;
	}

	public int get_out_depth() {
		return out_depth;
	}

	public String get_layer_type() {
		return layer_type;
	}

	public Vol get_out_act() {
		return out_act;
	}

	@Override
	public void set_in_act(Vol v) {
		this.in_act = v;
	}

	@Override
	public Vol get_in_act() {
		return this.in_act;
	}

	public ConvLayer(Definition opt) {

		// required
		this.out_depth = opt.filters;
		this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
		this.in_depth = opt.in_depth;
		this.in_sx = opt.in_sx;
		this.in_sy = opt.in_sy;

		// optional
		this.sy = opt.sy != 0 ? opt.sy : this.sx;
		this.stride = opt.stride != 0 ? opt.stride : 1; // stride at which we apply filters to input volume
		this.pad = opt.pad; // amount of 0 padding to add around borders of input volume
		this.l1_decay_mul = opt.l1_decay_mul;
		this.l2_decay_mul = opt.l2_decay_mul != 0.0 ? opt.l2_decay_mul : 1.0;

		// computed
		// note we are doing floor, so if the strided convolution of the filter doesnt
		// fit into the input
		// volume exactly, the output volume will be trimmed and not contain the
		// (incomplete) computed
		// final application.
		this.out_sx = (int) Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
		this.out_sy = (int) Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
		this.layer_type = "conv";

		// initializations
		double bias = opt.bias_pref;
		this.filters = new Vol[this.out_depth];
		for (int i = 0; i < this.out_depth; i++) {
			this.filters[i] = new Vol(this.sx, this.sy, this.in_depth, null);
		}
		this.biases = new Vol(1, 1, this.out_depth, bias);

	}
	@Override
	public Vol forward(Vol V, boolean is_training) {
	    this.in_act = V;
	    Vol A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

	    int V_sx = V.sx;
	    int V_sy = V.sy;
	    int xy_stride = this.stride;

	    var futures = Multithreading.newFutures();

	    for (int d = 0; d < this.out_depth; d++) {
	        final int depthIndex = d;
	        final Vol filter = this.filters[depthIndex];

	        Runnable r = () -> {
	            for (int ay = 0, y = -this.pad; ay < this.out_sy; ay++, y += xy_stride) {
	                for (int ax = 0, x = -this.pad; ax < this.out_sx; ax++, x += xy_stride) {
	                    double a = 0.0;

	                    for (int fy = 0; fy < filter.sy; fy++) {
	                        int oy = y + fy;
	                        if (oy < 0 || oy >= V_sy) continue;

	                        for (int fx = 0; fx < filter.sx; fx++) {
	                            int ox = x + fx;
	                            if (ox < 0 || ox >= V_sx) continue;

	                            for (int fd = 0; fd < filter.depth; fd++) {
	                                a += filter.w[((fy * filter.sx) + fx) * filter.depth + fd]
	                                    * V.w[((oy * V_sx) + ox) * V.depth + fd];
	                            }
	                        }
	                    }

	                    a += this.biases.w[depthIndex];
	                    A.set(ax, ay, depthIndex, a);
	                }
	            }
	        };

	        Multithreading.addAndExecuteFuture(null, futures, r);
	    }

	    Multithreading.waitForFutures(futures);
	    this.out_act = A;
	    return this.out_act;
	}
	public Vol forward1(Vol V, boolean is_training) {

		this.in_act = V;
//      Vol A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
//      int V_sx = V.sx |0;
//      int V_sy = V.sy |0;
//      int xy_stride = this.stride |0;
		Vol A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
		int V_sx = V.sx;
		int V_sy = V.sy;
		int xy_stride = this.stride;

		for (int d = 0; d < this.out_depth; d++) {
			Vol f = this.filters[d];
			int x = -this.pad;
			int y = -this.pad;
			var futures = Multithreading.newFutures();
		

				for (int ay = 0; ay < this.out_sy; y += xy_stride, ay++) { // xy_stride
					x = -this.pad;
					for (int ax = 0; ax < this.out_sx; x += xy_stride, ax++) { // xy_stride

						// convolve centered at this particular location
						double a = 0.0;
						for (int fy = 0; fy < f.sy; fy++) {
							int oy = y + fy; // coordinates in the original input array coordinates
							for (int fx = 0; fx < f.sx; fx++) {
								int ox = x + fx;
								if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
									for (int fd = 0; fd < f.depth; fd++) {
										// avoid function call overhead (x2) for efficiency, compromise modularity :(
										a += f.w[((f.sx * fy) + fx) * f.depth + fd]
												* V.w[((V_sx * oy) + ox) * V.depth + fd];
									}
								}
							}
						}
						a += this.biases.w[d];
						A.set(ax, ay, d, a);
					}
				}

		}
		this.out_act = A;

		return this.out_act;
	}
	@Override
	public void backward() {
	    Vol V = this.in_act;
	    int Vlen = V.w.length;

	    // Zero global gradients in-place instead of reallocating
	    java.util.Arrays.fill(V.dw, 0.0);
	    for (Vol filter : this.filters) java.util.Arrays.fill(filter.dw, 0.0);
	    java.util.Arrays.fill(this.biases.dw, 0.0);

	    int V_sx = V.sx;
	    int V_sy = V.sy;
	    int xy_stride = this.stride;
	    int V_depth = V.depth;

	    var futures = Multithreading.newFutures();

	    for (int d = 0; d < this.out_depth; d++) {
	        final int depthIndex = d;
	        final Vol filter = this.filters[depthIndex];

	        Runnable r = () -> {
	            double[] f_dw = filter.dw;
	            double[] V_dw_local = new double[Vlen]; // local accumulation
	            double bias_grad = 0.0;

	            double[] f_w = filter.w;
	            double[] V_w = V.w;

	            for (int ay = 0, y = -pad; ay < this.out_sy; ay++, y += xy_stride) {
	                for (int ax = 0, x = -pad; ax < this.out_sx; ax++, x += xy_stride) {

	                    double chain_grad = this.out_act.get_grad(ax, ay, depthIndex);
	                    bias_grad += chain_grad;

	                    for (int fy = 0; fy < filter.sy; fy++) {
	                        int oy = y + fy;
	                        if (oy < 0 || oy >= V_sy) continue;

	                        for (int fx = 0; fx < filter.sx; fx++) {
	                            int ox = x + fx;
	                            if (ox < 0 || ox >= V_sx) continue;

	                            int base_f = (fy * filter.sx + fx) * filter.depth;
	                            int base_v = (oy * V_sx + ox) * V_depth;

	                            for (int fd = 0; fd < filter.depth; fd++) {
	                                int ix_f = base_f + fd;
	                                int ix_v = base_v + fd;

	                                double vw = V_w[ix_v];
	                                double fw = f_w[ix_f];

	                                // No locking needed, this is local
	                                f_dw[ix_f] += vw * chain_grad;
	                                V_dw_local[ix_v] += fw * chain_grad;
	                            }
	                        }
	                    }
	                }
	            }

	            // Atomic add: synchronize only once per shared write
	            synchronized (V.dw) {
	                for (int i = 0; i < Vlen; i++) {
	                    V.dw[i] += V_dw_local[i];
	                }
	            }

	            synchronized (biases.dw) {
	                biases.dw[depthIndex] += bias_grad;
	            }
	        };

	        Multithreading.addAndExecuteFuture(null, futures, r);
	    }

	    Multithreading.waitForFutures(futures);
	}

	public void backward2() {
	    Vol V = this.in_act;
	    V.dw = global.zeros(V.w.length); // zero out gradients w.r.t input

	    // Zero gradients for filters and biases
	    for (Vol filter : this.filters) {
	        filter.dw = global.zeros(filter.w.length);
	    }
	    this.biases.dw = global.zeros(this.biases.w.length);

	    int V_sx = V.sx;
	    int V_sy = V.sy;
	    int xy_stride = this.stride;

	    var futures = Multithreading.newFutures();

	    for (int d = 0; d < this.out_depth; d++) {
	        final int depthIndex = d;
	        final Vol filter = this.filters[depthIndex];

	        Runnable r = () -> {
	            // Local gradient accumulators (to avoid thread collisions)
	            double[] local_filter_dw = new double[filter.w.length];
	            double[] local_input_dw = new double[V.w.length];
	            double local_bias_grad = 0.0;

	            for (int ay = 0, y = -this.pad; ay < this.out_sy; ay++, y += xy_stride) {
	                for (int ax = 0, x = -this.pad; ax < this.out_sx; ax++, x += xy_stride) {

	                    double chain_grad = this.out_act.get_grad(ax, ay, depthIndex);

	                    for (int fy = 0; fy < filter.sy; fy++) {
	                        int oy = y + fy;
	                        if (oy < 0 || oy >= V_sy) continue;

	                        for (int fx = 0; fx < filter.sx; fx++) {
	                            int ox = x + fx;
	                            if (ox < 0 || ox >= V_sx) continue;

	                            for (int fd = 0; fd < filter.depth; fd++) {
	                                int ix1 = ((oy * V_sx) + ox) * V.depth + fd;
	                                int ix2 = ((fy * filter.sx) + fx) * filter.depth + fd;

	                                local_filter_dw[ix2] += V.w[ix1] * chain_grad;
	                                local_input_dw[ix1] += filter.w[ix2] * chain_grad;
	                            }
	                        }
	                    }

	                    local_bias_grad += chain_grad;
	                }
	            }

	            // Merge local thread-safe updates into shared arrays
	            synchronized (filter) {
	                for (int i = 0; i < local_filter_dw.length; i++) {
	                    filter.dw[i] += local_filter_dw[i];
	                }
	            }

	            synchronized (V) {
	                for (int i = 0; i < local_input_dw.length; i++) {
	                    V.dw[i] += local_input_dw[i];
	                }
	            }

	            synchronized (biases) {
	                biases.dw[depthIndex] += local_bias_grad;
	            }
	        };

	        Multithreading.addAndExecuteFuture(null, futures, r);
	    }

	    Multithreading.waitForFutures(futures);
	}

	public void backward1() {

		Vol V = this.in_act;
		V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it
		for (Vol filter : this.filters) {
			filter.dw = global.zeros(filter.w.length);
		}
		this.biases.dw = global.zeros(this.biases.w.length);

		int V_sx = V.sx | 0;
		int V_sy = V.sy | 0;
		int xy_stride = this.stride | 0;

		var futures = Multithreading.newFutures();
		
		for (int d = 0; d < this.out_depth; d++) {
			
		
				
			
			Vol f = this.filters[d];
			int x = -this.pad | 0;
			int y = -this.pad | 0;
			for (int ay = 0; ay < this.out_sy; y += xy_stride, ay++) { // xy_stride
				x = -this.pad | 0;
				for (int ax = 0; ax < this.out_sx; x += xy_stride, ax++) { // xy_stride

					// convolve centered at this particular location
					double chain_grad = this.out_act.get_grad(ax, ay, d); // gradient from above, from chain rule
					for (int fy = 0; fy < f.sy; fy++) {
						int oy = y + fy; // coordinates in the original input array coordinates
						for (int fx = 0; fx < f.sx; fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
								for (int fd = 0; fd < f.depth; fd++) {
									// avoid function call overhead (x2) for efficiency, compromise modularity :(
									int ix1 = ((V_sx * oy) + ox) * V.depth + fd;
									int ix2 = ((f.sx * fy) + fx) * f.depth + fd;
									f.dw[ix2] += V.w[ix1] * chain_grad;
									V.dw[ix1] += f.w[ix2] * chain_grad;
								}
							}
						}
					}
					this.biases.dw[d] += chain_grad;
				}
			}
			
			
		}
		
		Multithreading.waitForFutures(futures);
	}

	public ParamsAndGrads[] getParamsAndGrads() {
		ParamsAndGrads[] response = new ParamsAndGrads[this.out_depth + 1];
		for (int i = 0; i < this.out_depth; i++) {
			response[i] = new ParamsAndGrads(this.filters[i].w, this.filters[i].dw, this.l1_decay_mul,
					this.l2_decay_mul);
		}
		response[this.out_depth] = new ParamsAndGrads(this.biases.w, this.biases.dw, 0.0, 0.0);

		return response;
	}



	@Override
	public void save(JSONObject obj) throws JSONException {
	    obj.put("layer_type", this.layer_type);
	    obj.put("sx", this.sx);
	    obj.put("sy", this.sy);
	    obj.put("stride", this.stride);
	    obj.put("pad", this.pad);
	    obj.put("in_depth", this.in_depth);
	    obj.put("out_depth", this.out_depth);
	    obj.put("out_sx", this.out_sx);
	    obj.put("out_sy", this.out_sy);
	    obj.put("l1_decay_mul", this.l1_decay_mul);
	    obj.put("l2_decay_mul", this.l2_decay_mul);

	    JSONArray filtersArray = new JSONArray();
	    for (Vol filter : this.filters) {
	        JSONArray wArray = new JSONArray();
	        for (double w : filter.w) wArray.put(w);
	        filtersArray.put(wArray);
	    }
	    obj.put("filters", filtersArray);

	    JSONArray biasesArray = new JSONArray();
	    for (double b : this.biases.w) biasesArray.put(b);
	    obj.put("biases", biasesArray);
	}

	@Override
	public void load(JSONObject obj) throws JSONException {
	    this.layer_type = obj.getString("layer_type");
	    this.sx = obj.getInt("sx");
	    this.sy = obj.getInt("sy");
	    this.stride = obj.getInt("stride");
	    this.pad = obj.getInt("pad");
	    this.in_depth = obj.getInt("in_depth");
	    this.out_depth = obj.getInt("out_depth");
	    this.out_sx = obj.getInt("out_sx");
	    this.out_sy = obj.getInt("out_sy");
	    this.l1_decay_mul = obj.getDouble("l1_decay_mul");
	    this.l2_decay_mul = obj.getDouble("l2_decay_mul");

	    JSONArray filtersArray = obj.getJSONArray("filters");
	    this.filters = new Vol[filtersArray.length()];
	    for (int i = 0; i < filtersArray.length(); i++) {
	        JSONArray wArray = filtersArray.getJSONArray(i);
	        this.filters[i] = new Vol(this.sx, this.sy, this.in_depth, 0.0);
	        for (int j = 0; j < wArray.length(); j++) {
	            this.filters[i].w[j] = wArray.getDouble(j);
	        }
	    }

	    JSONArray biasesArray = obj.getJSONArray("biases");
	    this.biases = new Vol(1, 1, this.out_depth, 0.0);
	    for (int i = 0; i < biasesArray.length(); i++) {
	        this.biases.w[i] = biasesArray.getDouble(i);
	    }
	}
}
